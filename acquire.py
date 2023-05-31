import os


def run_command(command):
    stream = os.popen(command)
    output = stream.read()
    print(output)


def get_data(amount):
    # run_command('wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-gal.fits.gz -P ./data')
    # run_command('wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-star.fits.gz -P ./data')

    f = open("spec-list.txt", "w")
    bands = ['r','g','i','u','z']

    for i in range(90,100+amount):
        for band in bands:
            f.write(f'frame-{band}-008162-6-0{i:03d}.fits.bz2\n')
        f.write(f'frame-irg-008162-6-0{i}.jpg\n')

    f.close()

    # run_command('wget -i spec-list.txt -r --no-parent -nd -B "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/" -P ./data')


get_data(20)