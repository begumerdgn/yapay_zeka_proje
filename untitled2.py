# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:01:26 2025

@author: begum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

dosya_yolu = r"C:\Users\begum\Downloads\GR12_Genetic Algorithm.xlsx"
df = pd.read_excel(dosya_yolu, sheet_name=1, header=None)
df = df[0].str.split(',', expand=True)
df = df.astype(float)
matris = df.to_numpy()
num_cities = len(df)
distances = np.zeros((num_cities, num_cities))

for i in range(num_cities):
    for j in range(num_cities):
        distances[i,j] = np.sqrt(np.sum((matris[i] - matris[j])**2))

class Member:
    def __init__(self,route):
        self.route = route
        self.fitness = 0

def create_route():
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def calculate_fitness(member):
    total_distance = 0
    for i in range(len(member.route)-1):
        a = member.route[i]
        b = member.route[i+1]
        total_distance += distances[a,b]
    total_distance += distances[member.route[-1],member.route[0]]
    member.fitness = 1 / total_distance
        
def crossover(parent1, parent2):
    size = len(parent1.route)
    start, end = sorted([random.randint(0,size-1) for _ in range(2)])
    child_route = [None] * size
    child_route[start:end] = parent1.route[start:end]
    
    pointer = 0
    for gene in parent2.route:
        if gene not in child_route:
            while child_route[pointer] is not None:
                pointer += 1
            child_route[pointer] = gene
    return Member(child_route)

"AYARLAR"
population_size = 100
generations =  100
mutation_rate=0.1
elitism_rate=0.5

def mutate(member, mutation_rate):
    for i in range(len(member.route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(member.route)-1)
            member.route[i], member.route[j] = member.route[j], member.route[i]

population = [Member(create_route()) for _ in range(population_size)]
best_member = None

mesafe_gecmisi = []
for g in range(generations):
    for m in population:
        calculate_fitness(m)   
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_member = population[0]
    current_best_distance = 1 / best_member.fitness
    mesafe_gecmisi.append(current_best_distance)
    next_gen = population[:int(elitism_rate * population_size)]
    
    while len(next_gen) < population_size:
        parent1 = random.choice(population[:5])
        parent2 = random.choice(population[:5])
        child = crossover(parent1, parent2)
        mutate(child,mutation_rate)
        next_gen.append(child)

    population = next_gen

best_route = [matris[i] for i in best_member.route]
best_distance = 1 / best_member.fitness


print("Best Distance:", best_distance)


plt.figure(figsize=(10, 6))
plt.plot(mesafe_gecmisi, color='#8e44ad', linewidth=2.5, label='En İyi Mesafe')
plt.fill_between(range(len(mesafe_gecmisi)), mesafe_gecmisi, max(mesafe_gecmisi), color='#8e44ad', alpha=0.1)
plt.title("Genetik Algoritma İyileşme Süreci", fontsize=14, fontweight='bold')
plt.xlabel("Jenerasyon (Zaman)", fontsize=12)
plt.ylabel("Toplam Mesafe (Maliyet)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
final_skor = mesafe_gecmisi[-1]
plt.scatter(len(mesafe_gecmisi)-1, final_skor, color='red', s=50, zorder=5)
plt.annotate(f'Minimum: {final_skor:.2f}', 
             xy=(len(mesafe_gecmisi)-1, final_skor), 
             xytext=(len(mesafe_gecmisi)-20, final_skor + (max(mesafe_gecmisi)-min(mesafe_gecmisi))*0.1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=11, fontweight='bold')

plt.show()

