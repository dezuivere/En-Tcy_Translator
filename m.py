import csv

def txt_to_csv(file1, file2, output_csv):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['en', 'Kn'])

        for line1, line2 in zip(lines1, lines2):
            csv_writer.writerow([line1.strip(), line2.strip()])

txt_to_csv('kn_tr.txt', 'en_tr.txt', 'output.csv')
