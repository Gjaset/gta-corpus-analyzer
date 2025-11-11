import pandas as pd

# Lista de personajes extraída del guion
personajes = [
    "CJ (Carl Johnson)", "Sweet Johnson", "Kendl Johnson", "Big Smoke", "Ryder", "Cesar Vialpando", "Catalina",
    "The Truth", "Woozie (Wu Zi Mu)", "Mike Toreno", "Jethro", "Dwaine", "Zero", "Maccer", "Kent Paul",
    "Ken Rosenberg (Rosie)", "Maria Latore", "Salvatore Leone", "Johnny Sindacco", "T-Bone Mendez", "Ran Fa Li",
    "OG Loc", "Madd Dogg", "Officer Frank Tenpenny", "Officer Eddie Pulaski", "Officer Jimmy Hernandez", "Emmet",
    "Denise Robinson", "Millie Perkins", "Helena Wankstein", "Barbara Schternvart", "Katie Zhan", "Michelle Cannes",
    "Tony (el loro de Rosie)", "Suzie", "Woozie’s Assistant", "Occupant (del casino)", "Homie 1", "Homie 2",
    "Homie 3", "Guard 1", "Guard 2", "Bodyguard", "Receptionist", "Pizza Co. Employee", "Hillbilly 1", "Hillbilly 2",
    "Hillbilly 3", "Hillbilly Woman", "Doorman 1", "Doorman 2", "Man 1", "Man 2", "Employee (casino)", "Freddy",
    "Vagos 1", "Hazer", "Jizzy B", "Berkley", "Hernandez", "Pulaski", "Paul", "Madd Dogg’s Manager",
    "Sweet’s Mom (Beverly Johnson, mención)", "Big Bear", "Little Weasel", "Kane", "B Dup", "Big Poppa", "Freddie",
    "Big Devil", "Little Devil", "Officer Carver", "Officer Pendelbury", "Officer Brown", "Officer Daniels",
    "Catalina’s Cousin", "Johnny Klebitz (cameo)", "Kenji Kasen (mención)"
]

# Crear DataFrame
df = pd.DataFrame(personajes, columns=["Personaje"])

# Guardar como CSV
csv_path = "/home/gjaset/Escritorio/python/personajes_gta_san_andreas.csv"
df.to_csv(csv_path, index=False)

csv_path
