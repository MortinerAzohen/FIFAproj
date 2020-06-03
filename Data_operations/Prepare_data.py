import pandas as pd


class DataOperations:
    def __init__(self,dataset_path = '../input/FutBinCards19.csv'):
        self.dataset_path = dataset_path

    def parseValue(self, x):
        x = str(x).replace('€', '')
        if ('M' in str(x)):
            x = str(x).replace('M', '')
            x = float(x) * 1000000
        elif ('K' in str(x)):
            x = str(x).replace('K', '')
            x = float(x) * 1000
        return float(x)

    def parsePosition(self,x, uniq):
        x = str(x).replace(x, str(uniq.index(x)))
        x = int(x)
        return x

    def import_data(self):
        string_details = []
        fifa_raw_dataset = pd.read_csv(self.dataset_path)
        seen = set()
        uniq = [x for x in fifa_raw_dataset['Position'] if x not in seen and not seen.add(x)]
        string_details.append(uniq)
        fifa_raw_dataset['Position'] = fifa_raw_dataset['Position'].apply(self.parsePosition, args=[uniq])
        uniq = [x for x in fifa_raw_dataset['Country'] if x not in seen and not seen.add(x)]
        string_details.append(uniq)
        fifa_raw_dataset['Country'] = fifa_raw_dataset['Country'].apply(self.parsePosition, args=[uniq])
        uniq = [x for x in fifa_raw_dataset['Club'] if x not in seen and not seen.add(x)]
        string_details.append(uniq)
        fifa_raw_dataset['Club'] = fifa_raw_dataset['Club'].apply(self.parsePosition, args=[uniq])
        uniq = [x for x in fifa_raw_dataset['WorkRate'] if x not in seen and not seen.add(x)]
        string_details.append(uniq)
        fifa_raw_dataset['WorkRate'] = fifa_raw_dataset['WorkRate'].apply(self.parsePosition, args=[uniq])
        fifa_raw_dataset['Price'] = fifa_raw_dataset['Price'].apply(self.parseValue)
        # wybrane atrybuty to przewidywania wartości pilakrza
        features = ['Price', 'WeakFoot', 'SkillsMoves', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending',
                    'Phyiscality', 'Position','Country','Club','WorkRate']
        fifa_dataset = fifa_raw_dataset[[*features]]
        selected_fifa_dataset = fifa_dataset[fifa_dataset['Price']>250]
        items = []
        items.append(string_details)
        items.append(selected_fifa_dataset)
        return items