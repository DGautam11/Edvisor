from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone

class Utils:
    @staticmethod
    def get_relative_time(date_str):
        now = datetime.now(timezone.utc)
        then = parser.parse(date_str)
        delta = relativedelta(now, then)

        if delta.days == 0:
            relative_time = "today"
        elif delta.days == 1:
            relative_time = "yesterday"
        elif delta.days < 7:
            relative_time = f"{delta.days} days ago"
        elif delta.weeks == 1:
            relative_time = "1 week ago"
        elif delta.weeks < 4:
            relative_time = f"{delta.weeks} weeks ago"
        elif delta.months == 1:
            relative_time = "1 month ago"
        elif delta.months < 12:
            relative_time = f"{delta.months} months ago"
        else:
            relative_time = f"{delta.years} year{'s' if delta.years > 1 else ''} ago"
        
        return relative_time
