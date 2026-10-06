import bisect
import logging
import math
import os
from copy import deepcopy
from typing import Dict, List, Literal, Tuple, Union

import numpy as np

import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings
from fdsreader.fds_classes import Mesh
from fdsreader.utils import Quantity
from fdsreader.utils.data import resample_mesh_axis

_HANDLED_FUNCTIONS = {np.mean: (lambda pl: pl.mean)}


def implements(np_function):
    """Decorator to register an __array_function__ implementation for Plot3Ds."""

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class SubPlot3D:
    """Subplot of a pl3d output for a single mesh.

    :ivar mesh: The mesh containing the data.
    :ivar times: Time steps for which this mesh has written data, matching the first axis of data. These can be
        fewer than the times of the parent :class:`Plot3D` while FDS is still running.
    """

    # Offset of the binary file to the end of the file header.
    _offset = fdtype.new((("i", 3),)).itemsize + fdtype.new((("i", 4),)).itemsize

    def __init__(self, mesh: Mesh, quantity_idx: int):
        self.file_paths: List[str] = list()  # Path to the binary data file for each time step
        self.times: List[float] = list()  # Time steps for which this mesh has written data
        self.mesh = mesh
        self._quantity_idx = quantity_idx

    def _add_timestep(self, time: float, file_path: str):
        time_idx = bisect.bisect(self.times, time)
        self.times.insert(time_idx, time)
        self.file_paths.insert(time_idx, file_path)

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the 3D data for each quantity of a single mesh.

        :returns: 4D numpy array with (t,x,y,z) as dimensions.
        """
        if not hasattr(self, "_data"):
            self._data = np.empty(shape=(len(self.file_paths),) + self.mesh.dimension.shape(), dtype=np.float32)
            dtype_data = fdtype.new((("f", self.mesh.dimension.size() * 5),))
            for t, file_path in enumerate(self.file_paths):
                with open(file_path, "rb") as infile:
                    infile.seek(self._offset)
                    self._data[t, :, :, :] = fdtype.read(infile, dtype_data, 1)[0][0].reshape(
                        self.mesh.dimension.shape() + (5,), order="F"
                    )[:, :, :, self._quantity_idx]
        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        if hasattr(self, "_data"):
            del self._data


class Plot3D(np.lib.mixins.NDArrayOperatorsMixin):
    """Plot3d file data container including metadata. Consists of multiple subplots, one for each
        mesh.

    :ivar times: All times for which data has been recorded.
    :ivar quantities: List with quantity objects containing information about recorded quantities
     calculated for this Plot3D with the corresponding short_name and unit.
    """

    def __init__(self, root_path: str):
        self._root_path = root_path
        self.times: List[float] = list()

        # List of all subplots this Plot3D consists of (one per mesh).
        self._subplots: Dict[str, SubPlot3D] = dict()

    def _add_subplot(self, filename: str, time: float, quantity: Quantity, quantity_idx: int, mesh: Mesh):
        self.quantity: Quantity = quantity
        if mesh.id not in self._subplots.keys():
            self._subplots[mesh.id] = SubPlot3D(mesh, quantity_idx)
        if not any(np.isclose(time, t) for t in self.times):
            bisect.insort(self.times, time)
        self._subplots[mesh.id]._add_timestep(time, os.path.join(self._root_path, filename))

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            _ = self._subplots[mesh.id].data

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubPlot` that contains data for the given mesh."""
        return self._subplots[mesh.id]

    @implements(np.mean)
    def mean(self) -> float:
        """Calculates the mean value of the whole Plot3D.

        :returns: The calculated mean value.
        """
        mean_sum = 0
        for subplot in self._subplots.values():
            mean_sum += np.mean(subplot.data)
        return mean_sum / len(self._subplots)

    @implements(np.std)
    def std(self) -> float:
        """Calculates the standard deviation for each quantity individually of the whole Plot3D.

        :returns: The calculated standard deviation.
        """
        mean = self.mean
        sum = np.sum([np.sum(np.power(subplot.data - mean, 2)) for subplot in self._subplots.values()])
        N = np.sum([subplot.data.size for subplot in self._subplots.values()])
        return np.sqrt(sum / N)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        for subplot in self._subplots.values():
            subplot.clear_cache()

    def to_global(
        self, masked: bool = False, fill: float = 0, return_coordinates: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[Literal["x", "y", "z"], np.ndarray]]]:
        """Creates a global numpy ndarray from all subplots.

        :param masked: Whether to apply the obstruction mask to the data or not.
        :param fill: The fill value to use for masked entries. Only used when masked=True.
        :param return_coordinates: If true, return the matching coordinate for each value on the generated grid.
        """
        if len(self._subplots) == 0:
            if return_coordinates:
                return np.array([]), {d: np.array([]) for d in ("x", "y", "z")}
            else:
                return np.array([])

        coord_min = {"x": math.inf, "y": math.inf, "z": math.inf}
        coord_max = {"x": -math.inf, "y": -math.inf, "z": -math.inf}
        for dim in ("x", "y", "z"):
            for subplot in self._subplots.values():
                co = subplot.mesh.coordinates[dim]
                coord_min[dim] = min(co[0], coord_min[dim])
                coord_max[dim] = max(co[-1], coord_max[dim])

        # The global grid will use the finest mesh as base and duplicate values of the coarser
        # meshes. Therefore, we first find the finest mesh and calculate the step size in each
        # dimension.
        step_sizes_min = {
            "x": coord_max["x"] - coord_min["x"],
            "y": coord_max["y"] - coord_min["y"],
            "z": coord_max["z"] - coord_min["z"],
        }
        step_sizes_max = {"x": 0, "y": 0, "z": 0}
        steps = dict()
        global_max = {"x": -math.inf, "y": -math.inf, "z": -math.inf}

        for dim in ("x", "y", "z"):
            for subplot in self._subplots.values():
                step_size = subplot.mesh.coordinates[dim][1] - subplot.mesh.coordinates[dim][0]
                step_sizes_min[dim] = min(step_size, step_sizes_min[dim])
                step_sizes_max[dim] = max(step_size, step_sizes_max[dim])
                global_max[dim] = max(subplot.mesh.coordinates[dim][-1], global_max[dim])

        for dim in ("x", "y", "z"):
            if step_sizes_min[dim] == 0:
                step_sizes_min[dim] = math.inf
                steps[dim] = 1
            else:
                steps[dim] = (
                    max(int(round((coord_max[dim] - coord_min[dim]) / step_sizes_min[dim])), 1) + 1
                )  # + step_sizes_max[dim] / step_sizes_min[dim]

        grid = np.full((self.n_t, steps["x"], steps["y"], steps["z"]), np.nan, dtype=np.float32)

        start_idx = dict()
        end_idx = dict()
        for subplot in self._subplots.values():
            subplot_data = self._data_on_global_times(subplot)
            if masked:
                mask = subplot.mesh.get_obstruction_mask(self.times)

            start_idx = {
                dim: int(round((subplot.mesh.coordinates[dim][0] - coord_min[dim]) / step_sizes_min[dim]))
                for dim in ("x", "y", "z")
            }
            end_idx = {
                dim: int(round((subplot.mesh.coordinates[dim][-1] - coord_min[dim]) / step_sizes_min[dim]))
                for dim in ("x", "y", "z")
            }

            # Map each axis onto the global grid in turn. Border points are dropped unless they lie on the
            # border of the simulation space, as all other border points appear twice where meshes overlap.
            # The global grid uses the finest mesh as base, so coarser meshes need their values repeated
            # along every axis to match its resolution.
            for axis, dim in enumerate(("x", "y", "z")):
                co = subplot.mesh.coordinates[dim]
                n_repeat = max(int(round((co[1] - co[0]) / step_sizes_min[dim])), 1)
                keep_last = bool(np.isclose(co[-1], global_max[dim]))
                if keep_last:
                    end_idx[dim] += 1
                subplot_data = resample_mesh_axis(subplot_data, axis + 1, n_repeat, keep_last)
                if masked:
                    mask = resample_mesh_axis(mask, axis + 1, n_repeat, keep_last)

            # If the slice should be masked, we set all cells at which an obstruction is in the
            # simulation space to the fill value set by the user
            if masked:
                subplot_data = np.where(mask, subplot_data, fill)

            grid[:, start_idx["x"] : end_idx["x"], start_idx["y"] : end_idx["y"], start_idx["z"] : end_idx["z"]] = (
                subplot_data.reshape(
                    (
                        self.n_t,
                        end_idx["x"] - start_idx["x"],
                        end_idx["y"] - start_idx["y"],
                        end_idx["z"] - start_idx["z"],
                    )
                )
            )

        if return_coordinates:
            coordinates = dict()
            for dim_index, dim in enumerate(("x", "y", "z")):
                coordinates[dim] = np.linspace(coord_min[dim], coord_max[dim], grid.shape[dim_index + 1])

        if return_coordinates:
            return grid, coordinates
        else:
            return grid

    def _data_on_global_times(self, subplot: SubPlot3D) -> np.ndarray:
        """Returns the data of a subplot for all time steps of the Plot3D, filling time steps for which the
        mesh has not written data (yet) with NaN.
        """
        t_global = [next(idx for idx, t in enumerate(self.times) if np.isclose(time, t)) for time in subplot.times]
        if len(set(t_global)) < len(t_global):
            raise ValueError(
                f"Mesh {subplot.mesh.id} has Plot3D data for several time steps that match the same time of the "
                "Plot3D, so they cannot be told apart."
            )
        if t_global == list(range(self.n_t)):
            return subplot.data.copy()

        logging.warning(
            "Mesh %s has Plot3D data for %d of %d time steps, the missing ones are filled with NaN.",
            subplot.mesh.id,
            len(subplot.times),
            self.n_t,
        )
        data = np.full((self.n_t,) + subplot.data.shape[1:], np.nan, dtype=subplot.data.dtype)
        data[t_global] = subplot.data
        return data

    @property
    def n_t(self) -> int:
        """Get the number of timesteps for which data was output."""
        return len(self.times)

    @property
    def subplots(self):
        """Returns a list with one SubPlot3D object per mesh."""
        return list(self._subplots.values())

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray."""
        raise UserWarning(
            "Plot3Ds can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containing all subplots is"
            " required, please use the 'to_global' method and use the returned numpy-array explicitly."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunction with a Plot3D as input.

        :returns: A new pl3d on which the ufunc has been applied.
        """
        if method != "__call__":
            logging.warning(
                "The %s method has been used which is not explicitly implemented. Correctness of"
                " results is not guaranteed. If you require this feature to be implemented please"
                " submit an issue on Github where you explain your use case.",
                method,
            )
        input_list = list(inputs)
        for i, inp in enumerate(inputs):
            if isinstance(inp, self.__class__):
                del input_list[i]

        new_pl3d = deepcopy(self)
        for subplot in self._subplots.values():
            subplot._data = ufunc(subplot.data, input_list[0], **kwargs)
        return new_pl3d

    def __array_function__(self, func, types, args, kwargs):
        """Method that will be called by numpy when using an array function with a Slice as input.

        :returns: The output of the array function.
        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
            # Note: this allows subclasses that don't override __array_function__ to handle Plot3Ds.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


# __array_function__ implementations
# ...
