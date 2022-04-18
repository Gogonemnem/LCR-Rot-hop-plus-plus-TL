import csv
import json
import tarfile
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests


def amazon(folder_path: str = "ExternalData"):
    # site http://deepyeti.ucsd.edu/jianmo/amazon/index.html
    amazon = r'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Electronics.json.gz'
    # amazon = r'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Magazine_Subscriptions.json.gz'

    print("Starting Download")
    r = requests.get(amazon)
    print("Download Finished")

    path = Path.cwd() / folder_path / amazon.split('/')[-1][:-8]

    print('Extracting file and writing csv file')
    with GzipFile(fileobj=BytesIO(r.content)) as tmp_json, open(r"../ExternalData\amazon.csv", 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['text', 'polarity'])
        writer.writeheader()

        for line in tmp_json:
            data = json.loads(line)
            if 'reviewText' not in data:
                continue

            if data['overall'] < 3:
                polarity = -1
            elif data['overall'] == 3:
                polarity = 0
            elif data['overall'] > 3:
                polarity = 1
            writer.writerow({"text": data['reviewText'], "polarity": polarity})
    print('csv file created')


def glove(glove_url: str, folder_path: str = "ExternalData"):
    path = Path.cwd() / folder_path

    print("Starting Download")
    r = requests.get(glove_url)
    print("Download Finished")

    # extracting the zip file contents
    print(f"Extracting files to {path}")
    file = ZipFile(BytesIO(r.content))
    file.extractall(path)


def semeval(year, folder_path: str = "ExternalData"):
    path = Path.cwd() / folder_path

    if year == 2015:
        urls = [
            r"http://metashare.ilsp.gr:8080/repository/download/b2ac9c0c198511e4a109842b2b6a04d751e6725f2ab847df88b19ea22cb5cc4a/",  # Restaurant train
            r"http://metashare.ilsp.gr:8080/repository/download/4ab77724612011e4acce842b2b6a04d73cf3cb586f894d30b3c8afdd98cfbdc8/",  # Laptop train
            r"http://metashare.ilsp.gr:8080/repository/download/d32aeb3e9ca011e4a350842b2b6a04d737ee004f7cdc428bbf1ad4bd67977d22/",  # Restaurant test
            r"http://metashare.ilsp.gr:8080/repository/download/a2bd9f229ca111e4a350842b2b6a04d7d9091e92fc7149f485037cb9e98809af/",  # Laptop test
            r"http://metashare.ilsp.gr:8080/repository/download/153796fc9ca211e4bf03842b2b6a04d73c1f9fdd8aff4c83884694f3ebf4e3b6/"  # Hotel test
        ]

    elif year == 2016:
        urls = [
            r"http://metashare.ilsp.gr:8080/repository/download/cd28e738562f11e59e2c842b2b6a04d703f9dae461bb4816a5d4320019407d23/",  # Restaurant train
            r"http://metashare.ilsp.gr:8080/repository/download/0ec1d3b0563211e58a25842b2b6a04d77d2f0983ccfa4936a25ddb821d46e220/",  # Laptop train
            r"http://metashare.ilsp.gr:8080/repository/download/42bd97c6d17511e59dbe842b2b6a04d721d1933085814d9daed8fbcbe54c0615/",  # Restaurant test
            r"http://metashare.ilsp.gr:8080/repository/download/03906e0ad17711e59dbe842b2b6a04d70f7103db654f4281a3f1a4ad39c05948/",  # Laptop test
        ]

    LOGIN_URL = "http://metashare.ilsp.gr:8080/login/"
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }

    with requests.Session() as s:
        r = requests.get(LOGIN_URL, headers=headers, verify=False)

        headers['cookie'] = '; '.join(
            [x.name + '=' + x.value for x in r.cookies])
        headers['content-type'] = 'application/x-www-form-urlencoded'
        csrf_token = r.cookies['csrftoken']
        login_data = {'csrfmiddlewaretoken': csrf_token, 'username': 'gogonemnem',
                      'password': 'Oversweet-Unissued1-Component', 'next': '/'}

        license_agreement()
        x = s.post(LOGIN_URL, headers=headers, data=login_data)

        for url in urls:
            form_data = {'licence_agree': 'on',
                         'in_licence_agree_form': 'True', 'licence': 'MS-NC-NoReD'}
            x = s.post(url, data=form_data)
            file = ZipFile(BytesIO(x.content))
            file.extractall(path)


def license_agreement():
    if input("Do you agree with the license agreement from MS-NC-NoReD? http://www.meta-net.eu/meta-share/meta-share-licenses/META-SHARE%20NonCommercial%20NoRedistribution-v%201.0.pdf ([Y]es/[N]o): ").lower() not in ('y', 'yes'):
        raise ValueError("You did not agree to the license agreement")


def yelp():
    from bs4 import BeautifulSoup
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }

    data = {
        "name": 'a',
        "email": 'a@gmail.com',
        "signature": "a",
        "terms_accepted": "y"
    }

    url = 'https://www.yelp.com/dataset/download'
    with requests.Session() as s:
        s.headers.update(headers)
        r = s.get(url, verify=True)
        soup = BeautifulSoup(r.content, 'html.parser')
        tag = soup.find(name="input", attrs={"name": "csrftok"})
        data["csrftok"] = tag["value"]

        r = s.post(url, data=data)
        soup = BeautifulSoup(r.content, 'html.parser')
        tag = soup.find(name="a", attrs={"class": "ybtn ybtn--primary"})

        print('Starting Download')
        file = s.get(tag['href'])
        print('Download Finished')

    print('Getting content')
    with tarfile.open(fileobj=BytesIO(file.content)) as tar:
        print('Extracting file & writing csv file')
        with tar.extractfile(tar.getmembers()[3]) as tmp_json, open(r"../ExternalData\yelp.csv", 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['text', 'polarity'])
            writer.writeheader()

            for line in tmp_json:
                data = json.loads(line)
                if 'text' not in data:
                    continue

                if data['stars'] < 3:
                    polarity = -1
                elif data['stars'] == 3:
                    polarity = 0
                elif data['stars'] > 3:
                    polarity = 1
                writer.writerow({"text": data['text'].replace("\n", " "), "polarity": polarity})
    print('csv file created')


if __name__ == '__main__':
    # glove("https://nlp.stanford.edu/data/glove.6B.zip")
    # glove("https://nlp.stanford.edu/data/glove.42B.300d.zip")
    # semeval(2015)
    semeval(2016)
    # yelp()
    # amazon()
    pass
