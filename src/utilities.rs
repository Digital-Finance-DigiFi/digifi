pub mod maths_utils;
pub mod time_value_utils;


use std::collections::HashMap;
use ndarray::Array1;
use csv::Reader;


pub const TEST_ACCURACY: f64 = 0.00000001;


#[derive(Clone)]
/// # Description
/// Type of parameter used in calculations.
pub enum ParameterType {
    Value { value: f64 },
    TimeSeries { values: Array1<f64> },
}


#[derive(Clone)]
/// # Description
/// Struct for generating time array.
pub enum Time {
    /// Creates a range of time steps given the provided settings.
    Range { initial_time: f64, final_time: f64, time_step: f64 },
    /// Uses provided array as the space of time steps.
    Sequence { times: Array1<f64> },
}

impl Time {
    /// # Description
    /// Generates the time array from the provided settings.
    pub fn get_time_array(&self) -> Array1<f64> {
        match self {
            Time::Range { initial_time, final_time, time_step } => {
                Array1::range(*initial_time, *final_time, *time_step)
            },
            Time::Sequence { times } => {
                times.clone()
            },
        }
    }
}


/// # Description
/// Asserts that the two arrays provided are of the same length, while also verifying that both arrays are of np.ndarray type.
/// 
/// # Input
/// - array_1: First array
/// - array_2: Second array
/// - array_1_name: Name of the first array, which will be printed in case of a panic
/// - array_1_name: Name of the second array, which will be printed in case of a panic
/// 
/// # Panics
/// - Panics if the length of arrays do not match
pub fn compare_array_len<T>(array_1: &Array1<T>, array_2: &Array1<T>, array_1_name: &str, array_2_name: &str) -> () {
    if array_1.len() != array_2.len() {
        panic!("The length of {} and {} do not coincide.", array_1_name, array_2_name);
    }
}


pub enum SampleData {
    CAPM,
    Portfolio,
    Stock,
}

impl SampleData {

    fn load_capm_data(&self) -> (Vec<String>, HashMap<String, Array1<f64>>) {
        let path: String = String::from("./sample_data/capm_data.csv");
        let mut reader = Reader::from_path(path).expect("Sample file is not found.");
        let mut dates: Vec<String> = Vec::<String>::new();
        let mut aapl: Vec<f64> = Vec::<f64>::new();
        let mut market: Vec<f64> = Vec::<f64>::new();
        let mut rf: Vec<f64> = Vec::<f64>::new();
        let mut smb: Vec<f64> = Vec::<f64>::new();
        let mut hml: Vec<f64> = Vec::<f64>::new();
        let mut rmw: Vec<f64> = Vec::<f64>::new();
        let mut cma: Vec<f64> = Vec::<f64>::new();
        for line in reader.records() {
            let line = line.expect("Error while parsing a line of the CSV file.");
            // CSV file order: Date, SMB, HML, RMW, CMA, RF, AAPL, Mkt
            dates.push(String::from(line.get(0).unwrap()));
            smb.push(line.get(1).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            hml.push(line.get(2).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            rmw.push(line.get(3).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            cma.push(line.get(4).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            rf.push(line.get(5).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            aapl.push(line.get(6).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            market.push(line.get(7).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
        }
        let capm_data: HashMap<String, Array1<f64>> = HashMap::from([(String::from("aapl"), Array1::from_vec(aapl)),
                                                                     (String::from("market"), Array1::from_vec(market)),
                                                                     (String::from("rf"), Array1::from_vec(rf)),
                                                                     (String::from("smb"), Array1::from_vec(smb)),
                                                                     (String::from("hml"), Array1::from_vec(hml)),
                                                                     (String::from("rmw"), Array1::from_vec(rmw)),
                                                                     (String::from("cma"), Array1::from_vec(cma)),]);
        (dates, capm_data)
    }

    fn load_portfolio_data(&self) -> (Vec<String>, HashMap<String, Array1<f64>>) {
        let path: String = String::from("./sample_data/portfolio_data.csv");
        let mut reader = Reader::from_path(path).expect("Sample file is not found.");
        let mut dates: Vec<String> = Vec::<String>::new();
        let mut bac: Vec<f64> = Vec::<f64>::new();
        let mut c: Vec<f64> = Vec::<f64>::new();
        let mut gs: Vec<f64> = Vec::<f64>::new();
        let mut jpm: Vec<f64> = Vec::<f64>::new();
        for line in reader.records() {
            let line = line.expect("Error while parsing a line of the CSV file.");
            // CSV file order: Date, BAC, C, GS, JPM
            dates.push(String::from(line.get(0).unwrap()));
            bac.push(line.get(1).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            c.push(line.get(2).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            gs.push(line.get(3).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            jpm.push(line.get(4).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
        }
        let portfolio_data: HashMap<String, Array1<f64>> = HashMap::from([(String::from("bac"), Array1::from_vec(bac)),
                                                                          (String::from("c"), Array1::from_vec(c)),
                                                                          (String::from("gs"), Array1::from_vec(gs)),
                                                                          (String::from("jpm"), Array1::from_vec(jpm)),]);
        (dates, portfolio_data)
    }

    fn load_stock_data(&self) -> (Vec<String>, HashMap<String, Array1<f64>>) {
        let path: String = String::from("./sample_data/stock_data.csv");
        let mut reader = Reader::from_path(path).expect("Sample file is not found.");
        let mut dates: Vec<String> = Vec::<String>::new();
        let mut high: Vec<f64> = Vec::<f64>::new();
        let mut low: Vec<f64> = Vec::<f64>::new();
        let mut open: Vec<f64> = Vec::<f64>::new();
        let mut close: Vec<f64> = Vec::<f64>::new();
        let mut volume: Vec<f64> = Vec::<f64>::new();
        let mut adj_close: Vec<f64> = Vec::<f64>::new();
        for line in reader.records() {
            let line = line.expect("Error while parsing a line of the CSV file.");
            // CSV file order: Date, High, Low, Open, Close, Volume, Adj Close
            dates.push(String::from(line.get(0).unwrap()));
            high.push(line.get(1).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            low.push(line.get(2).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            open.push(line.get(3).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            close.push(line.get(4).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            volume.push(line.get(5).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
            adj_close.push(line.get(6).unwrap().parse::<f64>().expect("Value from CSV cannot be converted to f64."));
        }
        let stock_data: HashMap<String, Array1<f64>> = HashMap::from([(String::from("high"), Array1::from_vec(high)),
                                                                      (String::from("low"), Array1::from_vec(low)),
                                                                      (String::from("open"), Array1::from_vec(open)),
                                                                      (String::from("close"), Array1::from_vec(close)),
                                                                      (String::from("volume"), Array1::from_vec(volume)),
                                                                      (String::from("adj_close"), Array1::from_vec(adj_close))]);
        (dates, stock_data)
    }

    pub fn load_sample_data(&self) -> (Vec<String>, HashMap<String, Array1<f64>>) {
        match self {
            SampleData::CAPM => { self.load_capm_data() },
            SampleData::Portfolio => { self.load_portfolio_data() },
            SampleData::Stock => { self.load_stock_data() },
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use crate::utilities::TEST_ACCURACY;

    #[test]
    fn unit_test_time_struct() -> () {
        use crate::utilities::Time;
        let time_1 = Time::Range { initial_time: 0.0, final_time: 1.0, time_step: 0.2 };
        let time_2 = Time::Sequence { times: Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8]) };
        let time_array_1: Array1<f64> = time_1.get_time_array();
        let time_array_2: Array1<f64> = time_2.get_time_array();
        assert!((time_array_1 - time_array_2).sum().abs() < TEST_ACCURACY);
    }

    #[test]
    fn unit_test_compare_vec_len() -> () {
        use crate::utilities::compare_array_len;
        let a: Array1<i32> = Array1::from_vec(vec![1, 2, 3]);
        let b: Array1<i32> = Array1::from_vec(vec![4, 5, 6]);
        compare_array_len(&a, &b, "a", "b");
    }

    #[test]
    fn unit_test_sample_capm() -> () {
        use crate::utilities::SampleData;
        let sample: SampleData = SampleData::CAPM;
        let (dates, capm_data) = sample.load_sample_data();
        assert_eq!(dates[2], String::from("2019-03-31"));
        assert_eq!(capm_data.get("rf").unwrap()[55], 0.0045000000000000005);
        assert_eq!(capm_data.get("hml").unwrap()[37], 0.0308);
    }

    #[test]
    fn unit_test_sample_portfolio() -> () {
        use crate::utilities::SampleData;
        let sample: SampleData = SampleData::Portfolio;
        let (dates, portfolio_data) = sample.load_sample_data();
        assert_eq!(dates[15], String::from("2023-03-07"));
        assert_eq!(portfolio_data.get("c").unwrap()[44], 47.88409423828125);
        assert_eq!(portfolio_data.get("jpm").unwrap()[23], 122.31652069091797);
    }

    #[test]
    fn unit_test_sample_stock() -> () {
        use crate::utilities::SampleData;
        let sample: SampleData = SampleData::Stock;
        let (dates, stock_data) = sample.load_sample_data();
        assert_eq!(dates[40], String::from("2010-03-03"));
        assert_eq!(stock_data.get("low").unwrap()[110], 8.834643363952637);
        assert_eq!(stock_data.get("adj_close").unwrap()[167], 7.665149211883545);
    }
}