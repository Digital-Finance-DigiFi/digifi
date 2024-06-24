use std::collections::HashMap;
use ndarray::Array1;


#[derive(Clone)]
/// # Description
/// Token data format used to define AMMLiquidityPool.
pub struct AMMToken {
    /// Token identifier/name
    pub id: String,
    /// Supply of token in the liquidity pool
    pub supply: f64,
    /// Lower bound for a possible fee
    pub fee_lower_bound: f64,
    /// Upper bound for a possible fee
    pub fee_upper_bound: f64,
}


/// # Description
/// Liquidity data for an AMM.
/// 
/// Characteristic Number = Token 1 Supply * Token 2 Supply
/// 
/// # LaTeX Formula
/// - \\textit{Characteristic Number} = S_{1}\\times S_{2}
pub struct AMMLiquidityPool {
    token_1: AMMToken,
    token_2: AMMToken,
    char_number: f64,
    tolerance: f64,
}

impl AMMLiquidityPool {
    /// # Description
    /// Creates a new AMMLiquidityPool instance.
    /// 
    /// # Input
    /// - token_1: Token 1 data
    /// - token_2: Token 2 data
    /// - char_number: Characteristic number
    /// - tolerance: Error margin for computing validating token supplies
    /// 
    /// # Panics
    /// - Panics if the characteristic number is not positive
    /// - Panics if the product of supplies of the two tokens do not equal to the characteristic number
    pub fn new(token_1: AMMToken, token_2: AMMToken, char_number: f64, tolerance: f64) -> Self {
        if char_number <= 0.0 {
            panic!("The argument char_number must be positive.")
        }
        if tolerance < ((token_1.supply * token_2.supply) - char_number).abs() {
            panic!("The argument char_number must be the product of supplies of the tokens.");
        }
        AMMLiquidityPool { token_1, token_2, char_number, tolerance }
    }

    pub fn token_1(&self) -> AMMToken {
        self.token_1.clone()
    }

    pub fn token_2(&self) -> AMMToken {
        self.token_2.clone()
    }

    pub fn char_number(&self) -> f64 {
        self.char_number
    }

    /// # Description
    /// Updates the state of the liquidity pool while preserving the characteristic number
    /// 
    /// # Input
    /// - token_1: Token 1 updated data
    /// - token_2: Token 2 updated data
    /// 
    /// # Panics
    /// - Panics if the product of updated supplies of the two tokens do not equal to the characteristic number
    /// - Panics if the wrong id for either token_1 or token_2 are provided
    pub fn update_token_supply(&mut self, token_1: AMMToken, token_2: AMMToken) -> () {
        if self.tolerance < ((token_1.supply * token_2.supply) - self.char_number).abs() {
            panic!("The argument char_number must be the product of supplies of the tokens.");
        }
        if self.token_1.id != token_1.id {
            panic!("Wrong token_1 id is provided.");
        }
        if self.token_2.id != token_2.id {
            panic!("Wrong token_2 id is provided.");
        }
        self.token_1 = token_1;
        self.token_2 = token_2;
    }
}


/// # Description
/// Transaction data used to pass transactions into AMM methods.
pub struct AMMTransactionData {
    /// Token identifier/name that is being purchased
    token_id: String,
    /// Number of tokens to purchase from exchange
    quantity: f64,
    /// Fee size as the percentage of transaction
    percent_fee: f64,
}

impl AMMTransactionData {
    pub fn new(token_id: String, quantity: f64, percent_fee: f64) -> Self {
        if quantity <= 0.0 {
            panic!("The argument quantity must be positive.");
        }
        if percent_fee < 0.0 {
            panic!("The argument percent_fee must be non-negative.");
        }
        AMMTransactionData { token_id, quantity, percent_fee }
    }

    pub fn token_id(&self) -> String {
        self.token_id.clone()
    }

    pub fn quantity(&self) -> f64 {
        self.quantity
    }

    pub fn percent_fee(&self) -> f64 {
        self.percent_fee
    }
}


/// # Description
/// Contains computational methods for an AMM with the liquidity pool given by:
/// 
/// Characteristic Number = Token 1 Supply * Token 2 Supply
/// 
/// # LaTeX Formula
/// - \\textit{Characteristic Number} = S_{1}\\times S_{2}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: https://doi.org/10.48550/arXiv.2106.14404
pub struct SimpleAMM {
    /// State of the liquidity pool to initiate the AMM with
    liquidity_pool: AMMLiquidityPool,
}

impl SimpleAMM {
    /// # Description
    /// Buy a quntity of a token from the AMM by submitting the buy order quoted in terms of the token to putchase.
    /// 
    /// Transaction includes fee as the percentage of the quantity purchased.
    /// 
    /// # Input
    /// - tx_data: Transaction data for AMM to process
    /// 
    /// # Output
    /// - quantity_to_sell: Amount of token that has to be sold to the AMM in exchange for the token being purchased
    /// - exchange_price: Exchange rate produced by the AMM
    /// - fee_in_purchased_token: Transaction fee that has to be paid quoted in quantity of purchased token (e.g., fee is 2.1 Purchased Tokens)
    /// 
    /// # Panics
    /// - Panics if the id of token in transaction does not match the id of any token in the liquidity pool
    /// - Panics if cannot fill the buy order due to lack of supply in the liquidity pool
    /// - Panics if the transaction fee is outside the fee range specified by the liquidity pool
    pub fn make_transaction(&mut self, tx_data: AMMTransactionData) -> HashMap<String, f64> {
        let mut token_1 = self.liquidity_pool.token_1();
        let mut token_2 = self.liquidity_pool.token_2();
        let token: AMMToken;
        let counterparty_token_supply: f64;
        if tx_data.token_id() == token_1.id {
            token = token_1.clone();
            counterparty_token_supply = token_2.supply;
        } else if tx_data.token_id() == token_2.id {
            token = token_2.clone();
            counterparty_token_supply = token_1.supply;
        } else {
            panic!("The token with identifier {} does not exist in the liquidity pool.", tx_data.token_id());
        }
        let tx_buy_size = tx_data.quantity() * (1.0 + tx_data.percent_fee());
        if token.supply < tx_buy_size {
            panic!("Not enough supply of token {} ({}) to fill in the buy order of {}.", token.id, token.supply, tx_buy_size);
        }
        if (tx_data.percent_fee() < token.fee_lower_bound) || (token.fee_upper_bound < tx_data.percent_fee()) {
            panic!("The argument percent_fee must be in the range [{}, {}].", token.fee_lower_bound, token.fee_upper_bound);
        }
        // Change in supply of token (y - delta_y)
        let updated_token_supply = token.supply - tx_buy_size;
        // Update supply of counterparty_token based on the characteristic number (x + delta_x = K/(y - delta_y))
        let updated_counterparty_token_supply = self.liquidity_pool.char_number() / updated_token_supply;
        // Determine amount of other_token that needs to be sold to AMM to fill the token buy order (delta_x)
        let dx = updated_counterparty_token_supply - counterparty_token_supply;
        // Exchange price (P = (x + delta_x)/(y - delta_y))
        let price = updated_counterparty_token_supply / updated_token_supply;
        // Fee quoted in terms of token
        let fee = tx_data.quantity() * tx_data.percent_fee() * price;
        // Update the liquidity pool
        if tx_data.token_id() == token_1.id {
            token_1.supply = updated_token_supply;
            token_2.supply = updated_counterparty_token_supply;
        } else {
            token_1.supply = updated_counterparty_token_supply;
            token_2.supply = updated_token_supply;
        }
        self.liquidity_pool.update_token_supply(token_1, token_2);
        let mut receipt: HashMap<String, f64> = HashMap::<String, f64>::new();
        receipt.insert(String::from("quantity_to_sell"), dx);
        receipt.insert(String::from("exchange_price"), price);
        receipt.insert(String::from("fee_in_purchased_token"), fee);
        receipt
    }

    /// # Description
    /// Generates points to plot the liquidity curve of the AMM.
    /// 
    /// # Input
    /// - n_points: Number of points to generate
    /// - token_1_start: Starting point of the x-axis
    /// - token_1_end: Final point of the x-axis
    /// 
    /// # Output
    /// - x: x-axis of the liquidity curve
    /// - y: y-axis of the liquidity curve
    pub fn get_liquidity_curve(&self, n_points: usize, token_1_start: f64, token_1_end: f64) -> HashMap<String, Array1<f64>> {
        let x: Array1<f64> = Array1::linspace(token_1_start, token_1_end, n_points);
        let y: Array1<f64> = self.liquidity_pool.char_number() / &x;
        let mut liquidity_curve: HashMap<String, Array1<f64>> = HashMap::<String, Array1<f64>>::new();
        liquidity_curve.insert(String::from("x"), x);
        liquidity_curve.insert(String::from("y"), y);
        liquidity_curve
    }
}


#[cfg(test)]
mod tests {

    #[test]
    fn unit_test_simple_amm_make_transaction() -> () {
        use crate::market_making::amm::{AMMToken, AMMLiquidityPool, AMMTransactionData, SimpleAMM};
        let token_1: AMMToken = AMMToken { id: String::from("BTC"), supply: 10.0, fee_lower_bound: 0.0, fee_upper_bound: 0.03  };
        let token_2: AMMToken = AMMToken { id: String::from("ETH"), supply: 1_000.0, fee_lower_bound: 0.0, fee_upper_bound: 0.03  };
        let liquidity_pool: AMMLiquidityPool = AMMLiquidityPool::new(token_1, token_2, 10_000.0, 0.00001);
        let tx_data: AMMTransactionData = AMMTransactionData::new(String::from("BTC"), 1.0, 0.01);
        let mut amm: SimpleAMM = SimpleAMM { liquidity_pool };
        let receipt = amm.make_transaction(tx_data);
        assert_eq!(*receipt.get("quantity_to_sell").unwrap(), 10_000.0/8.99 - 1_000.0);
        assert_eq!(*receipt.get("exchange_price").unwrap(), (10_000.0/8.99) / 8.99);
    }
}