use std::{cmp::Ordering, collections::BinaryHeap};

/// Max-heap element of neighbor with respect to a query (index)
pub struct QueryItem
{
    pub dist: f32,
    pub id: u32
}

impl PartialEq for QueryItem
{
    fn eq(&self, other: &Self) -> bool {return self.dist == other.dist;}
}

impl Eq for QueryItem {}

impl PartialOrd for QueryItem
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return self.dist.partial_cmp(&other.dist);
    }
}

impl Ord for QueryItem
{
    fn cmp(&self, other: &Self) -> Ordering {
        return self.partial_cmp(other).unwrap_or(Ordering::Equal);
    }
}

/// Extension trait to denote the sorting of QueryItem struct
pub trait SortQueryItemsExt {
    fn sort_query_into(self, idx_out: &mut [u32], dists_out: &mut [f32]);
}

impl SortQueryItemsExt for BinaryHeap<QueryItem>
{
    fn sort_query_into(self, idx_out: &mut [u32], dists_out: &mut [f32])
    {
        // Sorting neighbors
        let mut neighbors = self.into_vec();
        neighbors.sort();

        // Iterate through neighbor and write directly into buffering slice
        for (item, (id_out, d_out)) in neighbors.iter()
                                        .zip(idx_out.iter_mut().zip(dists_out.iter_mut())) {
            *id_out = item.id;
            *d_out = item.dist;                                
        }
    } 
}