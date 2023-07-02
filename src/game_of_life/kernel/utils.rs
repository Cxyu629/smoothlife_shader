pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    (start * (1.0 - t) + end * t).clamp(0.0, 1.0)
}
