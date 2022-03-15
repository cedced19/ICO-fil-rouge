CAPACITY = 100


class Customer:

    def __init__(self, customerID, positionX, positionY, demande) -> None:
        self.id = customerID
        self.positionX = positionX
        self.positionY = positionY
        self.demande = demande
        # TODO: Ajouter intervalle de temps
        self.intervalleTemps = None

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return str(self.id)


class Vehicule:

    def __init__(self, vehicleID, capacity) -> None:
        self.id = vehicleID
        self.maxCapacity = capacity
        self.actualCapacity = 0

    def addCustomer(self, customer: Customer):
        if (self.actualCapacity + customer.demande <= self.maxCapacity):
            self.actualCapacity += customer.demande
            return True
        return False


class Route:

    def __init__(self, vehicule: Vehicule, costMatrix) -> None:
        self.vehicule = vehicule
        self.customers = []
        self.coutTotal = 0
        self.costMatrix = costMatrix

    def addCustomer(self, customer: Customer):
        # Case is the first customer to be added.
        if (not self.customers):
            self.vehicule.addCustomer(customer)
            self.coutTotal += self.costMatrix[[0], customer.id]
            self.customers.append(customer)
            return True
        if (self.vehicule.addCustomer(customer)):
            self.coutTotal += self.costMatrix[self.customers[-1].id, customer.id]
            self.customers.append(customer)
            return True
        return False

    def calculateTotal(self):
        return self.coutTotal + self.costMatrix[self.customers[-1].id, 0]


class Solution:

    def __init__(self, costMatrix) -> None:
        self.routes = []
        # TODO Actually infinite vehicule
        self.vehicules = []
        self.vehiculesID = 0
        self.costMatrix = costMatrix

    def addCustomer(self, customer: Customer):
        if not self.routes:
            newVehicule = Vehicule(self.vehiculesID, CAPACITY)
            self.vehiculesID += 1
            newRoute = Route(newVehicule, self.costMatrix)
            newRoute.addCustomer(customer)
            self.routes.append(newRoute)
            return True
        else:
            if (self.routes[-1].addCustomer(customer)):
                return True
            # The route don't accept more customer we add another route
            newVehicule = Vehicule(self.vehiculesID, CAPACITY)
            self.vehiculesID += 1
            newRoute = Route(newVehicule, self.costMatrix)
            newRoute.addCustomer(customer)
            self.routes.append(newRoute)

    def calculateTotalCost(self):
        totalCost = 0
        for route in self.routes:
            totalCost += route.calculateTotal()
        return totalCost

    def __str__(self) -> str:
        string = f"ROUTES: {hex(id(self))} \n"
        for route in self.routes:
            string += f"Route: {route.customers} cost: {route.calculateTotal()}\n"
        string += f"Total cost: {self.calculateTotalCost()}"
        return string


# TEST:

if __name__ == "__main__":
    import numpy as np
    from random import shuffle

    matrice_example = \
        np.matrix([[0, 14, 18, 9, 5, 7],
                  [14, 0, 12, 4, 17, 1],
                  [18, 12, 0, 3, 2, 1],
                  [9, 4, 3, 0, 4, 8],
                  [5, 17, 2, 4, 0, 11],
                  [7, 1, 1, 8, 11, 0]])

    c1 = Customer(1, 0, 0, 10)
    c2 = Customer(2, 0, 0, 100)
    c3 = Customer(3, 0, 0, 25)
    c4 = Customer(4, 0, 0, 15)
    c5 = Customer(5, 0, 0, 5)

    customers = [c5, c4, c3, c2, c1]

    solution = Solution(matrice_example)

    while customers:
        customer = customers.pop()
        solution.addCustomer(customer)
    print(solution)

    print()
    customers = [c1, c2, c3, c4, c5]

    solution = Solution(matrice_example)

    while customers:
        customer = customers.pop()
        solution.addCustomer(customer)
    print(solution)

    print()
    customers = [c1, c2, c3, c4, c5]
    shuffle(customers)
    print(customers)
    solution = Solution(matrice_example)

    while customers:
        customer = customers.pop()
        solution.addCustomer(customer)
    print(solution)
