import matplotlib.pyplot as plt
import pandas as pd
from influxdb_client import InfluxDBClient


def fetch_data(token, org, url, bucket, measurement):
    client = InfluxDBClient(url=url, token=token, org=org, verify_ssl=False)
    query_api = client.query_api()

    query = f"""
    from(bucket: "{bucket}")
      |> range(start: -7d)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    result = query_api.query(org=org, query=query)
    client.close()
    return result


def prepare_dataframe(result):
    data = []
    for table in result:
        for record in table.records:
            data.append(
                {
                    "Time": record.get_time(),
                    "bt": record.values.get("bt"),
                    "bx_gsm": record.values.get("bx_gsm"),
                    "by_gsm": record.values.get("by_gsm"),
                    "bz_gsm": record.values.get("bz_gsm"),
                    "proton_speed": record.values.get("proton_speed"),
                    "proton_density": record.values.get("proton_density"),
                    "proton_temperature": record.values.get("proton_temperature"),
                }
            )

    df = pd.DataFrame(data)
    df.set_index("Time", inplace=True)
    df.dropna(inplace=True)
    return df


def save_csv(df, filename):
    df.to_csv(filename, index=False)


def plot_time_series(df):
    fields = [
        "bt",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "proton_speed",
        "proton_density",
        "proton_temperature",
    ]
    figures = []
    for field in fields:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[field], label=field)
        plt.title(f"Time Series for {field}")
        plt.xlabel("Time")
        plt.ylabel(field)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig = plt.gcf()  # Get the current figure before it's closed
        plt.close()  # Close plot to prevent it from displaying now
        figures.append(fig)
    return figures
