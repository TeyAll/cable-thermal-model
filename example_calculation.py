from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cable_thermal_model import (
    BondingType,
    CableLayer,
    CablePosition,
    CircuitType,
    ModelFactory,
    PipeFillType,
    PipeInputSchema,
    StaticEnvSoil,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitInAirFromCableIdInputSchema,
    CircuitInSoilFromCableIdInputSchema,
)

static_env = StaticEnvSoil()

# We add two cables to the environment at the same depth of 1 m, and 0.2 m apart from each other.
# The first circuit is a single circuit inside a pipe, while the second circuit is a linear circuit without a pipe.
# For the first circuit we do not specify the circuit type and bonding type,
# so they default to 'CircuitType.Single' and 'BondingType.NoBonding'.
circuit_input_1 = CircuitInSoilFromCableIdInputSchema(
    x=0.2,  # The second circuit lies 20 cm away from the first circuit
    y=-1,
    circuit_name="circuit_1",
    cable_id="GPLK 10/10 kV 3x185 Al",
    cable_source_file_path="data/example_cables.csv",
    pipe=PipeInputSchema(
        fill_type=PipeFillType.Water,
        outer_radius=0.08,  # 160 mm outer diameter
        sdr=11,
    ),
)
circuit_input_2 = CircuitInSoilFromCableIdInputSchema(
    x=0,
    y=-1,
    circuit_name="circuit_2",
    cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
    cable_source_file_path="data/example_cables.csv",
    circuit_type=CircuitType.Linear,
    bonding_type=BondingType.TwoSided,
)

# Add the circuits to the environment.
static_env.add_circuit_from_cable_id(circuit_input_1)
static_env.add_circuit_from_cable_id(circuit_input_2)

# In this case we use a scenario that lasts for a week with a step size of 1 hour.
START_DATE = datetime(2026, 1, 1)
DAYS_TO_SIMULATE = 7

scenario = pd.DataFrame(
    index=pd.date_range(start=START_DATE, end=START_DATE + pd.Timedelta(days=DAYS_TO_SIMULATE), freq="1h")
)

# We set the load of the first circuit to 200A for the entire week, while we model the load of the second circuit
# as a sinusoidal function that oscillates between 50 and 350A with a period of 24 hours.
scenario["load_circuit_1"] = 200.0
scenario["load_circuit_2"] = 200.0 + 150.0 * np.sin(2 * np.pi * scenario.index.hour / 24)
scenario["ambient_temperature"] = 15.0  # Ambient temperature is assumed to be constant at 15 degrees Celsius.
scenario["soil_thermal_resistivity"] = 0.75  # Soil thermal resistivity is assumed to be constant at 0.75 mK/W.
scenario["soil_thermal_capacity"] = 2.0e6  # Soil thermal capacity is assumed to be constant at 2e6 J/(m³K).

model = ModelFactory.create_model(static_env=static_env)
solution = model.run(scenario=scenario)
temperature_result = solution.result

# Plot the calculated conductor temperatures for both circuits.
# For the second circuit, we plot the temperature of the conductor in the linear center position.
plt.plot(
    temperature_result.index,
    temperature_result["circuit_1"][CablePosition.Single][CableLayer.Conductor],
    label="Circuit 1 Conductor Temperature (°C)",
)
plt.plot(
    temperature_result.index,
    temperature_result["circuit_2"][CablePosition.LinearCenter][CableLayer.Conductor],
    label="Circuit 2 Conductor Temperature (°C)",
)
plt.ylabel("Conductor Temperature (°C)")
plt.title("Calculated Conductor Temperatures for Circuit 1 and Circuit 2")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
plt.xticks(rotation=45)
plt.show()