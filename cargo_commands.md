# Test Commands:

**Run all unit tests:**
> cargo test

**Run all unit tests without hiding stdout:**
> cargo test -- --nocapture

**Run specific unit test:**
> cargo test test_fn_name -- --exact

**Run specific unit test without hiding stdout:**
> cargo test test_fn_name --exact --nocapture

**Run specific unit test inside a feature without hiding stdout:**
> cargo test test_fn_name --all-features -- --nocapture

**Run all doctests:**
> cargo test --doc


# Documentation Commands:

**Create documentation:**
> cargo doc --open

**Create documentation without dependencies:**
> cargo doc --open --no-deps


# Publishing Commands:

**Test publishing process:**
> cargo publish --dry-run

**Publish to Crates.io:**
> cargo publish