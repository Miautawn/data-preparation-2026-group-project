import pyarrow as pa

BASE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("userId", pa.int64()),
        pa.field("sport", pa.string()),
        pa.field("gender", pa.string()),
        pa.field("timestamp", pa.list_(pa.int64())),
        pa.field("heart_rate", pa.list_(pa.float64())),
        pa.field("longitude", pa.list_(pa.float64())),
        pa.field("latitude", pa.list_(pa.float64())),
        pa.field("derived_speed", pa.list_(pa.float64())),
        pa.field("derived_distance", pa.list_(pa.float64())),
        pa.field("time_elapsed", pa.list_(pa.int64())),
        pa.field("altitude", pa.list_(pa.float64())),
    ]
)

PREPROCESSED_BASE_SCHEMA = (
    BASE_SCHEMA.append(pa.field("time_elapsed_standardized", pa.list_(pa.float64())))
    .append(pa.field("heart_rate_standardized", pa.list_(pa.float64())))
    .append(pa.field("altitude_standardized", pa.list_(pa.float64())))
    .append(pa.field("derived_speed_standardized", pa.list_(pa.float64())))
    .append(pa.field("derived_distance_standardized", pa.list_(pa.float64())))
    .append(pa.field("userId_idx", pa.int64()))
    .append(pa.field("sport_idx", pa.int64()))
    .append(pa.field("gender_idx", pa.int64()))
)
