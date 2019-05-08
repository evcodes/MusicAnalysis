import pandas

URL = "https://docs.google.com/spreadsheets/d/1S5C-SUvfR6F9zL_gqH6UmHIiwvB8Jo8hES8MzzEwvxk/export?format=csv&gid=834450549"
df = pandas.read_csv(URL)

print(df.head())
