import pandas as pd
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io.votable import parse
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt


# assumes that the last part of each target's ID is unique
def shorten_id(ID):
    s = ID.split("+") if "+" in id else ID.split("-")
    return s[-1]


table = Table.read('data/christ_table.csv', format='ascii.csv')

coordinates = SkyCoord(table['RA_2000'], table['Dec_2000'], unit=(u.hourangle, u.deg))

table['RA_2000_deg'] = coordinates.ra.deg
table['Dec_2000_deg'] = coordinates.dec.deg
table["ShortID"] = table["ID"].apply()
table.write('data/christ_table_degrees.csv', format='ascii.csv', overwrite=True)

fig, ax = plt.subplots()
ax.set(xlabel="RA, deg", ylabel="Dec, deg", title="Christopolous et al. targets, downselected")
scatter = ax.scatter(table["RA_2000_deg"], table["Dec_2000_deg"], label=table["ID"], c=table["Period"], cmap="inferno")
cbar = plt.colorbar(scatter, label='Period')

plt.legend()
plt.show()

# christ_table
# df_2 = pd.read_csv("data/christopoulou_position.csv")
# christ_df["ID"] = christ_df["ID"].astype(pd.StringDtype())
# df_2["ID"] = df_2["ID"].astype(pd.StringDtype())
# df_2["RA_2000"] = df_2["RA_2000"].apply(lambda c: Angle(c,unit=u.hour))
# df_2["Dec_2000"] = df_2["Dec_2000"].apply(lambda c: Angle(c,unit=u.deg))
# as
# print(christ_df.columns)
# print(df_2.columns)
# print(df_2.info())
# print(christ_df.info())
# christ_df = christ_df.join(df_2, on="ID", lsuffix="_1", how="inner")
# print(christ_df)
# print(christ_df.columns)


