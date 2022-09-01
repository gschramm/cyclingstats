from datetime import datetime, timezone
import pytz
from pathlib import Path

import pydantic
import fitparse
import geopy.distance
import numpy as np

from tzwhere import tzwhere


class Ride(pydantic.BaseModel):
    name: str
    distance: float
    moving_time: float
    avg_speed: float
    max_speed: float
    ascent: float
    descent: float
    datetimes: list[datetime]
    coordinates: list[tuple[float, float]]
    local_timezone: str
    major_version: int = 1
    minor_version: int = 1


def parse_fit_file(fname: Path) -> Ride:
    fitfile = fitparse.FitFile(str(fname))

    time = []
    dist = []
    speed = []
    alt = []
    coords = []

    # factor to convert from semicircles to degrees
    semi_circ_to_deg = (180 / 2**31)

    # Get all data messages that are of type record
    for record in fitfile.get_messages('record'):
        field_names = [x.name for x in record.fields]

        if 'position_long' in field_names:
            time.append(record.get('timestamp').value)
            if 'enhanced_altitude' in field_names:
                alt_field_name = 'enhanced_altitude'
            else:
                alt_field_name = 'altitude'
            alt_unit = record.get(alt_field_name).units
            if alt_unit == 'm':
                alt.append(record.get(alt_field_name).value / 1000)
            else:
                raise NotImplementedError

            if 'enhanced_speed' in field_names:
                speed_field_name = 'enhanced_speed'
            else:
                speed_field_name = 'speed'
            speed_unit = record.get(speed_field_name).units
            if speed_unit == 'm/s':
                speed.append(record.get(speed_field_name).value * 3.6)
            else:
                raise NotImplementedError

            coords.append(
                (record.get('position_lat').value * semi_circ_to_deg,
                 record.get('position_long').value * semi_circ_to_deg))

    sm_kernel = np.ones(15) / 15

    alt = np.array(alt)
    alt_sm = np.convolve(alt, sm_kernel, mode='same')
    speed = np.array(speed)

    # calculate difference between time tags
    time = np.array(time)
    time_delta = np.array([x.seconds for x in (time[1:] - time[:-1])])

    dist = geopy.distance.geodesic(*coords).km

    # calculate ascent and descent
    alt_diff = alt_sm[1:] - alt_sm[:-1]
    ascent = alt_diff[alt_diff >= 0].sum()
    descent = alt_diff[alt_diff < 0].sum()

    # calculate moving time
    t_mov = time_delta[speed[:-1] > 4].sum() / 60
    avg_speed = dist / (t_mov / 60)

    # number of datapoints
    num_p = len(coords)
    # number of elements to store
    num_e = min(20, len(coords))

    sub_coords = coords[::(num_p // num_e)][:num_e]
    sub_times = time.tolist()[::(num_p // num_e)][:num_e]

    # convert sub_times from UTC to local time zone
    tz = tzwhere.tzwhere(forceTZ=True)
    loc_tz = tz.tzNameAt(*sub_coords[0], forceTZ=True)

    sub_times = [
        utc_dt.replace(tzinfo=timezone.utc).astimezone(
            tz=pytz.timezone(loc_tz)) for utc_dt in sub_times
    ]

    ride = Ride(name=fname.stem.split('__')[1],
                distance=dist,
                moving_time=t_mov,
                avg_speed=avg_speed,
                max_speed=speed.max(),
                ascent=ascent,
                descent=descent,
                datetimes=sub_times,
                coordinates=sub_coords,
                local_timezone=loc_tz)

    return ride