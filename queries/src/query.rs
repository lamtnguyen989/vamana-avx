use std::cmp::Ordering;

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