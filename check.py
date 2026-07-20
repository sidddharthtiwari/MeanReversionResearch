from src.data.loader import *

print("Available sectors:")
print(available_sectors())

print("\nBank:")
print(load_sector("bank").head())

print("\nBANK:")
print(load_sector("BANK").head())

print("\n Bank :")
print(load_sector(" Bank ").head())

print("\nMetadata:")
print(load_sector_metadata("bank").head())