import csv

path = './data/industry_load'
for i in range(5):
    with open(path + '/load%s.txt' % i, 'r') as infile, open(path + '/load%s.csv' % i, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(";") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)