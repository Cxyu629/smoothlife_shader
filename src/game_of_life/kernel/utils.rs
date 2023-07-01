
pub fn linear_from_points((x1, y1): (f32, f32), (x2, y2): (f32, f32)) -> impl Fn(f32) -> f32 {
    let gradient = (y2 - y1) / (x2 - x1);

    move |x| gradient * (x - x1) + y1
}