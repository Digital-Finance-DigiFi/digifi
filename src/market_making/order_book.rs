/// # Description
/// The difference between the best bid and best ask quotes divided by their sum.\n
/// Volume Imbalance = (Best Bid Volume - Best Ask Volume) / (Best Bid Volume + Best Ask Volume)
/// 
/// # Input
/// - best_bid_volume: Volume of the best bid price on the bid side of the order book
/// - best_ask_volume: Volume of the best ask price on the ask side of the order book
/// 
/// # Output
/// - Volume imbalance of the order book
/// 
/// # LaTeX Formula
/// - Imb_{t} = \\frac{V^{b}_{t}-V^{a}_{t}}{V^{b}_{t}+V^{a}_{t}}
/// 
/// # Links
/// - Wikipedia: N/A
/// - Original Source: https://davidsevangelista.github.io/post/basic_statistics_order_imbalance/#:~:text=The%20Order%20Book%20Imbalance%20is,at%20the%20best%20ask%2C%20respectively.
pub fn volume_imbalance(best_bid_volume: i32, best_ask_volume: i32) -> f64 {
    let best_bid_volume = best_bid_volume as f64;
    let best_ask_volume = best_ask_volume as f64;
    (best_bid_volume - best_ask_volume) / (best_bid_volume + best_ask_volume)
}


// TODO: Add order book algorithm


#[cfg(test)]
mod tests {

    #[test]
    fn unit_test_volume_imbalance() -> () {
        use crate::market_making::order_book::volume_imbalance;
        let volume_imbalance = volume_imbalance(10_000, 10_100);
        assert_eq!(volume_imbalance, -100.0 / 20_100.0);
    }
}