import csv

import sys
import csv
csv.field_size_limit(2147483647)   # Uncomment this to be able to run code

def filter_data(filenames):
    guardian_articles = []
    counter = 1
    
    timeperiod = ['2016-07',
                '2016-08',
                '2016-09',
                '2016-10',
                '2016-11',
                '2016-12',
                '2017-01',
                '2017-02',
                '2017-03',
                '2017-04',
                '2017-05',
                '2017-06']

    monthly_articles = {}
    for filename in filenames:
        print(filename)
        with open(filename, mode='r', encoding="utf8") as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                for line in reader:
                    newspaper = line[3]
                    year_and_month = line[5][0:7]
                    if newspaper =='Guardian' and year_and_month in timeperiod:
                        print(year_and_month)
                        print(counter)
                        #input(line)
                        timeish = line[5][0:7]
                        #input(timeish)
                        try:
                            monthly_articles[timeish] += 1
                        except KeyError:
                            monthly_articles[timeish] = 1
                        guardian_articles.append(line)
                        counter += 1

    for key in sorted(monthly_articles):
        print(key, monthly_articles[key])

    with open('Guardian - Filtered' + '.csv', mode='w', encoding="utf8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            counter = 1
            for article in guardian_articles:
                print(counter)
                counter += 1
                try:
                    csv_writer.writerow(article)
                except UnicodeEncodeError:
                    print('\n, article')



def main():
    filenames = ('articles3.csv', 'articles2.csv', 'articles1.csv')
    filter_data(filenames)


if __name__=='__main__':
    main()