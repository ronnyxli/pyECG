
import wfdb
import pdb


db_name = 'mitdb'

# get list of records
records = wfdb.get_record_list(db_name, records='all')

# loop all records
for record in records:
    if db_name == 'mitdb':

        # read data
        data = wfdb.rdsamp(record, pb_dir=db_name)
        pdb.set_trace()
        data[0]
        fs = data[1]['fs']

        # read annotations
        ann = wfdb.rdann(record, extension='atr', pb_dir=db_name)
        ann.symbol
        ann.sample

        pdb.set_trace()
    else:
        print('Database unknown')

pdb.set_trace()


# get sampling rate and save signals in dict
fs = record[1]['fs']
data = {}
for n in range(0, len(record[1]['sig_name'])):
    data[record[1]['sig_name'][n]] = record[0][:,n]
