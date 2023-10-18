from astropy.io.votable import parse
votable = parse("data/marsh_catalog_votable.vot")
votable = votable.get_first_table()
print(votable)
print(votable.fields)