from influxdb import InfluxDBClient
import pandas as pd

#from write_data import co231
from write_data import co231, co231_pr, h239_pr, h2s234_pr, h240_pr

client = InfluxDBClient(host="localhost", port=8086)

client.create_database("Street")
client.switch_database("Street")

co231(client)
# h239(client)
# h2s234(client)
# h240(client)
# no2232(client)
# o3235(client)
# pm10236(client)
# pm25237(client)
# so2233(client)
# t238(client)

# здесь загрузить предсказанные данные
# co231_pr(client)
# h239_pr(client)
# h2s234_pr(client)
h240_pr(client)
