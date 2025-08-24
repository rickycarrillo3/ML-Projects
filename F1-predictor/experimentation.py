import fastf1

if __name__ == "__main__":
    sesh_obj = fastf1.get_session(2021, 'Monaco', 'Q')
    sesh_obj.load(telemetry=False, weather=False, laps=False)
    leclerc = sesh_obj.get_driver('LEC')
    print(f"First name: {leclerc['FirstName']}, Last name: {leclerc['LastName']}")

