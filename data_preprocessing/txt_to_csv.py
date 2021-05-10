import csv

def txt_to_csv(base_path):
    industry_load_original_path = base_path + 'data/industry_load'
    for i in range(5):
        with open(industry_load_original_path + '/load%s.txt' % i, 'r') as infile, open(industry_load_original_path + '/load%s.csv' % i, 'w') as outfile:
            stripped = (line.strip() for line in infile)
            lines = (line.split(";") for line in stripped if line)
            writer = csv.writer(outfile)
            writer.writerows(lines)
