from crossfire import Client


client = Client(email="evaldocunhaf@gmail.com", password="Heitoreeu290803") # credentials are optional, the default are the environment variables
# client.states()
df = client.cities(format="df")
df = client.occurrences('813ca36b-91e3-4a18-b408-60b27a1942ef', format='df')
df.to_csv('recife_occurrences.csv', index=False)
# print(df[df['name'] == 'Recife'])
# print(df.head())
