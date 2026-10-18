import pandas as pd

from src.data import clean_events, make_room_key


def test_room_key_ignores_punctuation_and_case():
    df = pd.DataFrame({"Building": ["60", "60.0"], "Floor/Room": ["G.57", "g57"]})
    assert list(make_room_key(df)) == ["60_G57", "60_G57"]


def test_room_key_separates_buildings():
    df = pd.DataFrame({"Building": ["60", "69"], "Floor/Room": ["G.20", "G.20"]})
    assert make_room_key(df).nunique() == 2


def test_clean_events_keeps_highest_id_per_timestamp():
    t = pd.Timestamp("2025-03-03 10:00", tz="Australia/Melbourne")
    ev = pd.DataFrame({"floorspaceid": ["a", "a", "a"],
                       "time": [t, t, t + pd.Timedelta("1min")],
                       "id": [1, 2, 3], "headcount": [5, 6, 7]})
    out = clean_events(ev)
    assert len(out) == 2
    assert out.loc[0, "headcount"] == 6
    
