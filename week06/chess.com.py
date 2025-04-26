import requests
import json

username = "yoonhero"

headers = {
    'User-Agent': f'my-profile-tool/1.2 (username: {username}; contact: (contact email))',
    'Accept-Encoding': 'gzip',
    'Accept': 'application/json, text/plain, */**'
}
response = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives", headers=headers)
if response.status_code == 200:
    archives = response.json().get("archives", [])


pgns = ""
for archive in archives:
    print(f"Requesting {archive}")
    # response = requests.get(f"{archive}/pgn")
    response = requests.get(f"{archive}", headers=headers)
    pgn = [game["pgn"] for game in response.json()["games"]]
    pgns += "\n\n".join(pgn) + "\n\n"
total_plays = len(pgns.split("\n\n")) - 1

print(f"{total_plays=}")

with open("my_chess.pgn", "w") as f:
    f.write(pgns)