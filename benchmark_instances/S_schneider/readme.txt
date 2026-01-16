Large scale instances for the CLRP.

The distances between nodes is given by the euclidean distance multiplied by 100 and rounded up to the next integer.

The JSON files contain the following keys:

"customers": List of customer nodes. Every entry has the following keys:
	"demand": Demand of the customer.
	"index": Unique index of this customer amongst all customer and depot nodes.
	"x": X coordinate in the plane.
	"y": Y coordinate in the plane.

"depots": List of depot facility nodes. Every entry has the following keys:
	"capacity": Capacity constraint of the depot.
	"costs": Fixed opening costs of the depot.
	"index": Unique index of this depot amongst all customer and depot nodes.
	"x": X coordinate in the plane.
	"y": Y coordinate in the plane.

"name": Name of the instance.
"type": Type of the instance.
"vehicle_capacity": Capacity constraint per vehicle.
"vehicle_costs": Fixed costs for every utilized vehicle.