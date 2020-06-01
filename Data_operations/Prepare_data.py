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
        fifa_raw_dataset = pd.read_csv(self.dataset_path)
        seen = set()
        uniq = [x for x in fifa_raw_dataset['Position'] if x not in seen and not seen.add(x)]
        fifa_raw_dataset['Position'] = fifa_raw_dataset['Position'].apply(self.parsePosition, args=[uniq])
        fifa_raw_dataset['Price'] = fifa_raw_dataset['Price'].apply(self.parseValue)
        # wybrane atrybuty to przewidywania wartości pilakrza
        features = ['Price', 'WeakFoot', 'SkillsMoves', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending',
                    'Phyiscality', 'Position']
        fifa_dataset = fifa_raw_dataset[[*features]]
        return fifa_dataset