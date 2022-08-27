import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import fitparse

from utils import Ride, parse_fit_file
from plotting import ride_to_figure, bokeh_cycling_stats


class RideStats:

    def __init__(self, fnames: list[Path]) -> None:
        self.fnames = fnames
        self.rides = []

        for fname in self.fnames:
            # commuting file that was not renamed
            if not fname.stem.startswith('20'):
                fitfile = fitparse.FitFile(str(fname))
                for i, record in enumerate(fitfile.get_messages('record')):
                    if i == 0:
                        date = record.get('timestamp').value.strftime('%Y%m%d')
                        break

                new_filename = fname.parent / f'{date}__commute.FIT'

                i = 2
                while new_filename.exists():
                    new_filename = new_filename.parent / f'{new_filename.stem}_{i}{new_filename.suffix}'

                print(f'moving {fname} {new_filename}')
                fname.rename(new_filename)
                fname = new_filename

            preprocessed_fname = fname.with_suffix('.json')

            if not preprocessed_fname.exists():
                print(f'parsing {fname}')
                ride = parse_fit_file(fname)
                with open(preprocessed_fname, 'w') as f:
                    f.write(ride.json())
            else:
                print(f'loading {preprocessed_fname}')
                ride = Ride.parse_file(preprocessed_fname)

            screenshot_fname = fname.with_suffix('.png')
            if not screenshot_fname.exists():
                fig = ride_to_figure(ride)
                fig.savefig(screenshot_fname)
                print(f'writing {screenshot_fname}')
                plt.close(fig)

            self.rides.append(ride)

    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame([x.__dict__ for x in self.rides])
        return df


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    rs = RideStats(
        sorted(
            list(
                Path(config['fileio']['datapath']).rglob(
                    config['fileio']['pattern']))))

    bokeh_cycling_stats(rs.df, 'cycling_stats.html')
