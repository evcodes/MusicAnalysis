from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup


def get_top_songs(year_start, year_end):
    delta = year_end - year_start

    output = []

    for i in range(0, delta):
        for j in range(1, 13):

            print(j)
            year = year_start + i
            month = str(j).zfill(2)
            day = 15

            date = "" + str(year) + "-" + str(month) + "-" + str(day)

            url = "http://www.billboard.com/charts/hot-100/" + date
            print(url)
            result = requests.get(url)
            soup = BeautifulSoup(result.content, 'html.parser')

            title_tag_name = soup.find_all("span",
                                           class_="chart-list-item__title-text")
            artist_tag_name = soup.find_all("div",
                                            class_="chart-list-item__artist")
            rank_tag_name = soup.find_all("div", class_="chart-list-item__rank")

            for l in range(0, len(title_tag_name)):
                row = "date, rank, song, artist"
                for k in range(0, len(title_tag_name[l].contents)):

                    if (len(artist_tag_name[l].findChildren()) >= 1):
                        artist = \
                        artist_tag_name[l].findChildren("a")[0].contents[
                            0].strip('\n')
                    else:
                        artist = str(artist_tag_name[l].contents[k].strip('\n'))

                    row = date + "," + str(
                        rank_tag_name[l].contents[k].strip("\n")) + "," + str(
                        title_tag_name[j].contents[k].strip(
                            "\n")) + "," + artist + '\n'
                output.append(row)
    with open('top_songs_per_month_per_year.csv', 'w') as file:
        for m in output:
            file.write(m.strip('\n'))
            file.write('\n')


get_top_songs(1998, 2018)