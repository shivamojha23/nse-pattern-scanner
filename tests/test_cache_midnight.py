import datetime
from zoneinfo import ZoneInfo
from freezegun import freeze_time
from db_cache import get_utc_now, _parse_meta_date

def test_assertions():
    # Freeze time to 1:00 AM IST on July 14, 2026.
    # This is 19:30 UTC on July 13, 2026.
    print("--- Running Direct Assertions ---")
    
    with freeze_time("2026-07-13 19:30:00", tz_offset=0):
        # 1. Test get_utc_now to IST conversion
        utc_now = get_utc_now()
        ist_date = utc_now.astimezone(ZoneInfo("Asia/Kolkata")).date()
        
        expected_date = datetime.date(2026, 7, 14)
        print(f"Frozen UTC time: {utc_now}")
        print(f"Calculated IST date: {ist_date}")
        assert ist_date == expected_date, f"Expected {expected_date}, got {ist_date}"
        print("[PASS] IST date calculation properly shifts to the next day across the midnight boundary.")
        
        # 2. Test _parse_meta_date backward compatibility
        # We simulate what the DB might hold
        old_format_str = "2026-07-10"
        new_format_str = "2026-07-09T22:30:00+00:00" # This is 4:00 AM IST on July 10
        
        parsed_old = _parse_meta_date(old_format_str)
        parsed_new = _parse_meta_date(new_format_str)
        
        print(f"Old format ('{old_format_str}') parsed to: {parsed_old}")
        print(f"New format ('{new_format_str}') parsed to: {parsed_new}")
        
        assert parsed_old == datetime.date(2026, 7, 10), "Failed to parse old YYYY-MM-DD format"
        assert parsed_new == datetime.date(2026, 7, 10), "Failed to parse new ISO format to correct IST date"
        print("[PASS] _parse_meta_date successfully parses both old and new formats.")
        
        print("\nAll direct assertions passed! [OK]")

if __name__ == "__main__":
    test_assertions()
