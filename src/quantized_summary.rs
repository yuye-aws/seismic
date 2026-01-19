use serde::{Deserialize, Serialize};

use crate::{DataType, SpaceUsage, SparseDataset};

use crate::elias_fano::EliasFano;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct QuantizedSummary {
    n_summaries: usize,
    d: usize,
    offsets: EliasFano,
    summaries_ids: Box<[u16]>, // There cannot be more than 2^16 summaries
    values: Box<[f32]>,
}

impl SpaceUsage for QuantizedSummary {
    fn space_usage_byte(&self) -> usize {
        SpaceUsage::space_usage_byte(&self.n_summaries)
            + SpaceUsage::space_usage_byte(&self.d)
            + SpaceUsage::space_usage_byte(&self.offsets)
            + SpaceUsage::space_usage_byte(&self.summaries_ids)
            + SpaceUsage::space_usage_byte(&self.values)
    }
}

impl QuantizedSummary {
    #[must_use]
    pub fn distances_iter(&self, query_components: &[u16], query_values: &[f32]) -> DistancesIter {
        DistancesIter::new(self, query_components, query_values)
    }
}

impl<T> From<SparseDataset<T>> for QuantizedSummary
where
    T: DataType,
{
    /// # Panics
    /// Panics if the number of summmaries is more than 2^16 (i.e., u16::MAX)
    fn from(dataset: SparseDataset<T>) -> QuantizedSummary {
        assert!(
            dataset.len() <= u16::MAX as usize,
            "Number of summaries cannot be more than 2^16"
        );

        // TODO: if dim is big it may be better to use a HashMap to map the components to the summaries
        let mut inverted_pairs = Vec::with_capacity(dataset.dim());
        for _ in 0..dataset.dim() {
            inverted_pairs.push(Vec::new());
        }

        for (doc_id, (components, values)) in dataset.iter().enumerate() {
            for (&c, &value) in components.iter().zip(values) {
                inverted_pairs[c as usize].push((value.to_f32().unwrap(), doc_id));
            }
        }

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len());
        let mut summaries_ids: Vec<u16> = Vec::with_capacity(dataset.nnz());
        let mut float_values = Vec::with_capacity(dataset.nnz());

        offsets.push(0);
        for ip in inverted_pairs.iter() {
            float_values.extend(ip.iter().map(|(v, _)| *v));
            summaries_ids.extend(ip.iter().map(|(_, id)| *id as u16));
            offsets.push(summaries_ids.len())
        }

        Self {
            n_summaries: dataset.len(),
            d: dataset.dim(),
            offsets: EliasFano::from(&offsets),
            summaries_ids: summaries_ids.into_boxed_slice(),
            values: float_values.into_boxed_slice(),
        }
    }
}

pub struct DistancesIter {
    current: usize,
    distances: Vec<f32>,
}

impl DistancesIter {
    fn new(summaries: &QuantizedSummary, query_components: &[u16], query_values: &[f32]) -> Self {
        let mut accumulator = vec![0_f32; summaries.n_summaries];

        for (&qc, &qv) in query_components.iter().zip(query_values) {
            if qc as usize >= summaries.d {
                break;
            }
            let current_offset = summaries.offsets.select(qc as usize).unwrap();
            let next_offset = summaries.offsets.select((qc + 1) as usize).unwrap();

            if next_offset - current_offset == 0 {
                continue;
            }
            let current_summaries_ids = &summaries.summaries_ids[current_offset..next_offset];
            let current_values = &summaries.values[current_offset..next_offset];

            for (&s_id, &val) in current_summaries_ids.iter().zip(current_values) {
                accumulator[s_id as usize] += val * qv;
            }
        }

        Self {
            current: 0,
            distances: accumulator,
        }
    }
}

impl Iterator for DistancesIter {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.distances.len() {
            let current = self.current;
            self.current += 1;
            Some(self.distances[current])
        } else {
            None
        }
    }
}
