#this is it. Let's do this !

slowo = 'jjffffffaaaaaaa'


dlugosc = len(slowo)
litera = ''
max_litera = ''
aktualny_licznik = 1
max_licznik = 1

for idx in range(dlugosc-1):
    litera = slowo[idx]
    if slowo[idx] == slowo[idx+1]:
        aktualny_licznik+=1
    if aktualny_licznik > max_licznik:
        max_licznik = aktualny_licznik
        max_litera = litera
    if slowo[idx] != slowo[idx+1]:
        aktualny_licznik=1
        # max_licznik = 0

print(max_litera,max_licznik)