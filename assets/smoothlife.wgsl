@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

// struct Params {
//     kernel_size: u32,
//     random_float: f32,
// }

// @group(0) @binding(1)
// var <uniform> params: Params;

// @group(0) @binding(1) var random_seed: f32;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

fn rand(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }
fn noise(p: f32) -> f32 {
  let fl = floor(p);
  let fc = fract(p);
  return mix(rand(fl), rand(fl + 1.), fc);
}

fn rand2(n: vec2<f32>) -> f32 {
  return fract(sin(dot(n, vec2<f32>(12.9898, 4.1414))) * 43758.5453);
}

fn noise2(n: vec2<f32>) -> f32 {
  let d = vec2<f32>(0., 1.);
  let b = floor(n);
  let f = smoothstep(vec2<f32>(0.), vec2<f32>(1.), fract(n));
  return mix(mix(rand2(b), rand2(b + d.yx), f.x), mix(rand2(b + d.xy), rand2(b + d.yy), f.x), f.y);
}

fn circle(n: vec2<f32>, size: f32) -> f32 {
    if (length(n - vec2<f32>(size)) < size / 2.0) {
        return 0.0;
    } else if (length(n - vec2<f32>(size)) < size) {
        return 1.0;
    } else {
        return 0.0;
    }
}

fn permute4(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn fade2(t: vec2<f32>) -> vec2<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2<f32>) -> f32 {
    var Pi: vec4<f32> = floor(P.xyxy) + vec4<f32>(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4<f32>(0., 0., 1., 1.);
    Pi = Pi % vec4<f32>(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4<f32> = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2<f32> = vec2<f32>(gx.x, gy.x);
    var g10: vec2<f32> = vec2<f32>(gx.y, gy.y);
    var g01: vec2<f32> = vec2<f32>(gx.z, gy.z);
    var g11: vec2<f32> = vec2<f32>(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4<f32>(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2<f32>(fx.x, fy.x));
    let n10 = dot(g10, vec2<f32>(fx.y, fy.y));
    let n01 = dot(g01, vec2<f32>(fx.z, fy.z));
    let n11 = dot(g11, vec2<f32>(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2<f32>(n00, n01), vec2<f32>(n10, n11), vec2<f32>(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let random_number = randomFloat(invocation_id.y * num_workgroups.x + invocation_id.x);
    let alive = random_number > 0.75;
    // let color = vec4<f32>(circle(vec2<f32>(f32(invocation_id.x), f32(invocation_id.y)), 25.0));

    let zoom = 35.0;
    
    let color = vec4<f32>(perlinNoise2(vec2<f32>(f32(invocation_id.x)/zoom, f32(invocation_id.y)/zoom)));


    textureStore(texture, location, color);
}

const ra: f32 = 21.0;
const ri: f32 = 7.0;
const b1: f32 = 0.278;
const b2: f32 = 0.365;
const d1: f32 = 0.267;
const d2: f32 = 0.445;
const an: f32 = 0.028;
const am: f32 = 0.147;
const b: f32 = 1.0;
const timestep: f32 = 0.1;

fn wrap(coords: vec2<i32>) -> vec2<i32> {
    let dimensions: vec2<i32> = textureDimensions(texture);
    return vec2<i32>(i32(fract(f32(coords.x) / f32(dimensions.x)) * f32(dimensions.x)), i32(fract(f32(coords.y) / f32(dimensions.y)) * f32(dimensions.y)));
}

fn calculate_m(location: vec2<i32>) -> f32 {
    var sum: f32 = 0.0;
    var area: f32 = 0.0;
    for (var dx: f32 = -ri-b; dx <= ri+b; dx += 1.0) {
        for (var dy: f32 = -ri-b; dy <= ri+b; dy += 1.0) {
            let l = length(vec2(dx, dy));
            if (l < (ri - b/2.0)) {
                area += 1.0;
                sum += textureLoad(texture, wrap(location + vec2<i32>(i32(dx), i32(dy)))).x;
            } else if (l <= (ri + b/2.0)) {
                let weight = (ri + b/2.0 - l) / b;
                area += weight;
                sum += weight * textureLoad(texture, wrap(location + vec2<i32>(i32(dx), i32(dy)))).x;
            } 
        }
    }

    return sum / area;
}

fn calculate_n(location: vec2<i32>) -> f32 {
    var sum: f32 = 0.0;
    var area: f32 = 0.0;
    for (var dx: f32 = -ra-b; dx <= ra+b; dx += 1.0) {
        for (var dy: f32 = -ra-b; dy <= ra+b; dy += 1.0) {
            let l = length(vec2(dx, dy));
            if (l <= (ri - b/2.0)) {}
            else if (l < (ri + b/2.0)) {
                let weight = (l - ri + b/2.0) / b;
                area += weight;
                sum += weight * textureLoad(texture, wrap(location + vec2<i32>(i32(dx), i32(dy)))).x;
            } else if (l < (ra - b/2.0)) {
                area += 1.0;
                sum += textureLoad(texture, wrap(location + vec2<i32>(i32(dx), i32(dy)))).x;
            }  else if (l <= (ra + b/2.0)) {
                let weight = (ra + b/2.0 - l) / b;
                area += weight;
                sum += weight * textureLoad(texture, wrap(location + vec2<i32>(i32(dx), i32(dy)))).x;
            } 
        }
    }

    return sum / area;
}

fn logistic_threshold(x: f32, x0: f32, alpha: f32) -> f32 {
    return 1.0 / (1.0 + exp(- 4.0 / alpha * (x - x0)));
}

fn logistic_interval(x: f32, a: f32, b: f32, alpha: f32) -> f32 {
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha));
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return (1.0 - t) * a + t * b;
}

fn new_s(n: f32, m: f32) -> f32 {
    let aliveness = logistic_threshold(m, 0.5, am);


    let threshold1 = lerp(b1, d1, aliveness);
    let threshold2 = lerp(b2, d2, aliveness);

    let new_aliveness = logistic_interval(n, threshold1, threshold2, an);

    return clamp(new_aliveness, 0.0, 1.0);
}

fn clamp(x: f32, low: f32, high: f32) -> f32 {
    if (x < low) {return low;}
    if (x > high) {return high;}
    return x;
}

fn is_alive(location: vec2<i32>, offset_x: i32, offset_y: i32) -> i32 {
    let dimensions: vec2<i32> = textureDimensions(texture);
    let sum = (location + vec2<i32>(offset_x, offset_y));
    let new_location: vec2<i32> = vec2<i32>(i32(fract(f32(sum.x) / f32(dimensions.x)) * f32(dimensions.x)), i32(fract(f32(sum.y) / f32(dimensions.y)) * f32(dimensions.y)));
    let value: vec4<f32> = textureLoad(texture, new_location);
    return i32(value.x);
}

fn count_alive(location: vec2<i32>) -> i32 {
    return is_alive(location, -1, -1) +
           is_alive(location, -1,  0) +
           is_alive(location, -1,  1) +
           is_alive(location,  0, -1) +
           is_alive(location,  0,  1) +
           is_alive(location,  1, -1) +
           is_alive(location,  1,  0) +
           is_alive(location,  1,  1);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));


    // let n_alive = count_alive(location);

    // var alive: bool;
    // if (n_alive == 3) {
    //     alive = true;
    // } else if (n_alive == 2) {
    //     let currently_alive = is_alive(location, 0, 0);
    //     alive = bool(currently_alive);
    // } else {
    //     alive = false;
    // }

    // let color = vec4<f32>(f32(alive));


    let m = calculate_m(location);
    let n = calculate_n(location);

    let current = clamp(textureLoad(texture, location).x, 0.0, 1.0);

    let color = vec4<f32>( current + timestep * (2.0 * clamp(new_s(n,m), 0.0, 1.0) - 1.0));

    storageBarrier();

    textureStore(texture, location, color);
}