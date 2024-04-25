enum CompoundingType {
    Continuous,
    Periodic,
}

struct Compounding {
    rate: f64,
    compounding_type: CompoundingType,
    compounding_frequency: i32,
}