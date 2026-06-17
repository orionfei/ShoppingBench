# Unconstructable Clean Cases

- hard_index=86 line_no=298 sample_id=train_line_000298: target 5098503796 for "dart flight with medium shaft length" only hit with answer/title token `no2`; user-derived queries did not find it within pages 1-5/top10.
- hard_index=117 line_no=408 sample_id=train_line_000408: target 3466047941 for "casual red leather sling bag" only hit with answer/title brand `Aldo`; user-derived queries did not find it.
- hard_index=137 line_no=459 sample_id=train_line_000459: target 3607657940 for "short-sleeve cotton men's T-shirt in off white" only hit with answer/title terms like `graphic tee`/`Macbeth`; user-derived queries did not find it.
- hard_index=152 line_no=494 sample_id=train_line_000494: target 4328063885 for "black striped shirt with a shirt collar in size small" was not found by user-derived queries; only tested hit required answer/title sleeve wording (`half sleeve`).
- hard_index=163 line_no=526 sample_id=train_line_000526: target 4948904433 for same-shop basketball jersey size M was not found with user-derived global or shop_id=500219 queries; hits required answer/title terms such as `OMS` or `short sleeve`.
- hard_index=169 line_no=559 sample_id=train_line_000559: target 3305858391 for "stainless steel whistling kettle for gas range" was not found by user-derived queries; hits required answer/title capacity/brand terms such as `3.0L` or `Micromatic`.
- hard_index=172 line_no=572 sample_id=train_line_000572: target 4342860270 for "jump starter" was not found by plain user query; hits required added answer/spec terms such as `12v`, `1500a`, `Noco`, or `Ultrasafe`.
- hard_index=200 line_no=654 sample_id=train_line_000654: target 4026774086 for "soap for sensitive skin that's white" was not found by user-derived queries; hit required answer/product-line term `Glow Xpert`.
- hard_index=209 line_no=677 sample_id=train_line_000677: target 4788312750 for "women's shorts in yellow, size 3XL" was not found by user-derived queries; hit required answer/title style term `printed` or brand.
- hard_index=212 line_no=681 sample_id=train_line_000681: target 3396333427 for "dark brown eyebrow pencil" was not found by user-derived queries; hit required answer/title brand term `SANSAN`.
