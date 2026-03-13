"""
A class to use to simulate a federated learning environment locally. This class
adheres to the ProtocolFedLearning protocol of this project. This class represents
a single client in a federated learning environment.
- FedLearnSimulationGateway: A helper class that allows multiple instances to concurrently send data around.
- LocalFedLearningSimulator: The simulator class, representing a single client in a federated learning environment.
- LocalFedLearningSimulationWrapper: A wrapper class to simulate a federated learning environment locally. Contains
    multiple instances of LocalFedLearningSimulator and prepares the folder for running them.
"""
from threading import Lock
import os
import time
import shutil
from typing import Any, Optional, List
from .protocolfedlearningclass import ProtocolFedLearning

WAITING_TIME = 0.1

### Helpers
class FedLearnSimulationGatewayDataPacket:
    """
    A helper class containing one data package that is sent around in the FedLearnSimulationGateway.
    """
    def __init__(self, direction: str, memo: Optional[str], data: Any):
        self.direction = direction
        self.memo = memo
        self.data = data
        if self.direction not in ["to_coordinator", "to_clients"]:
            raise ValueError("Invalid direction, must be 'to_coordinator' or 'to_clients'")

class FedLearnSimulationGateway:
    """
    A helper class that allows multiple instances to concurrently send data around.
    """

    def __init__(self, num_clients: int):
        self._shared_dict: dict[str, List[FedLearnSimulationGatewayDataPacket]] = {}
            # Key is the client_id, value are the data packets of this client
        self._lock = Lock()
        self.num_clients = num_clients

    def send_to_coordinator(self, client_id: str, data: Any, memo: Optional[str] = None):
        with self._lock:
            if not client_id in self._shared_dict:
                self._shared_dict[client_id] = []
            self._shared_dict[client_id].append(
                FedLearnSimulationGatewayDataPacket(
                    direction="to_coordinator",
                    memo=memo,
                    data=data))

    def gather_data_for_coordinator(self, memo: Optional[str] = None) -> List[Any]:
        with self._lock:
            data_packets = []
            for _, packets in self._shared_dict.items():
                for packet in packets:
                    if packet.direction == "to_coordinator" and memo == packet.memo:
                        # if memo is not used it's None anyways, None == None
                        data_packets.append(packet.data)
            if len(data_packets) == self.num_clients:
                # done gathering, reset for this memo
                for client_id, packets in self._shared_dict.items():
                    self._shared_dict[client_id] = [packet for packet in packets if not (packet.direction == "to_coordinator" and memo == packet.memo)]
                return data_packets
            else:
                # not done gathering yet, return empty list
                return []

    def broadcast_to_clients(self, data: Any, memo: Optional[str] = None):
        with self._lock:
            for client_id in self._shared_dict.keys():
                self._shared_dict[client_id].append(
                    FedLearnSimulationGatewayDataPacket(
                        direction="to_clients",
                        memo=memo,
                        data=data))

    def await_data(self, n: int, client_id: str, direction: str, memo: Optional[str]) -> Optional[Any]:
        """
        More specialized than gather_data plus can also be called by clients plus ignores the
        direction.
        Finds n datapoints with memo x sent to this specific client.
        """
        with self._lock:
            data_packets = []
            client_packets = self._shared_dict.get(client_id, [])
            for packet in client_packets:
                if packet.memo == memo and packet.direction == direction:
                    data_packets.append(packet.data)
            if len(data_packets) == n:
                # done gathering, reset for this memo
                self._shared_dict[client_id] = [packet for packet in client_packets if not (packet.memo == memo and packet.direction == direction)]
                return data_packets
            return None


### Client class
class LocalFedLearningSimulator(ProtocolFedLearning):
    """
    The simulator class, representing a single client in a federated learning environment.
    """
    def __init__(self,
                 is_coordinator: bool,
                 client_id: int,
                 num_clients: int,
                 inputfolder: str,
                 outputfolder: str,
                 gateway: FedLearnSimulationGateway):
        """
        The constructor, the following parameters are required:

        Args:
            is_coordinator: Boolean variable, if True the this client
                represents the coordinator. False otherwise.
            client_id: The id of the client
            num_clients: The total number of clients in the federated learning
                environment. This is needed to correctly gather data,
                specifically to know how many data packets to wait for
            inputfolder: The path to the input folder containing the data for this client
            outputfolder: The path to the output folder to store the results
            gateway: The federated learning gateway instance to facilitate communication.
        """
        self.coordinator = is_coordinator
        self.client_id = client_id
        if client_id == 'global':
            raise ValueError("The client_id 'global' is reserved for global data")
        self.num_clients = num_clients
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        self.gateway = gateway
        self.previously_awaited_data = None
        self.round_counter = 0

    @property
    def is_coordinator(self):
        return self.coordinator

    @property
    def id(self):
        return self.client_id

    def send_data_to_coordinator(self,
                                 data,
                                 send_to_self=True,
                                 use_smpc=False,
                                 use_dp=False,
                                 memo=None):
        print(f"Client {self.client_id} sending data to coordinator")
        self.gateway.send_to_coordinator(client_id=str(self.client_id), data=data, memo=memo)

    def gather_data(self,
                    is_json: bool=False,
                    use_smpc: bool=False,
                    use_dp: bool=False,
                    memo: Optional[Any]=None):
        if not self.coordinator:
            raise ValueError("Only the coordinator can gather data")
        # wait for enough data to arrive in a loop
        while True:
            data_packets = self.gateway.gather_data_for_coordinator(memo=memo)
            if len(data_packets) == self.num_clients:
                return data_packets
            time.sleep(WAITING_TIME)

    def broadcast_data(self,
                       data: Any,
                       send_to_self: bool = True,
                       use_dp: bool = False,
                       memo: Optional[Any] = None) -> None:
        if not self.coordinator:
            raise ValueError("Only the coordinator can broadcast data")
        print(f"Broadcasting data to all clients")
        self.gateway.broadcast_to_clients(data=data, memo=memo)

    def await_data(self,
                     n: int = 1,
                     unwrap: bool = True,
                     is_json: bool = False,
                     use_dp: bool = False,
                     use_smpc: bool = False,
                     memo: Optional[Any] = None):
        """
        Wait for n data packets. Could be called by clients or the coordinator.
        """
        while True:
            direction = "to_clients" if n==1 else "to_coordinator"
                # we just assume that any single package waiting is from a send to a client
                # while any more packages should only ever be send to the coordinator (aggregator)
            data = self.gateway.await_data(n=n, client_id=str(self.client_id), direction=direction, memo=memo)
            if data is not None:
                if unwrap and len(data) == 1:
                    return data[0]
                return data
            time.sleep(WAITING_TIME)

### Wrapper class
class LocalFedLearningSimulationWrapper:
    """
    A wrapper class to simulate a federated learning environment locally.
    This class is used to create multiple instances of LocalFedLearningSimulator
    and run them concurrently.
    """
    def __init__(self,
                 clientfolders: List[str],
                 outputfolders: List[str],
                 generic_dir: Optional[str]) -> None:
        """
        The following arguments are required:

        Args:
            clientfolders: A list of paths to the client folders containing the data
            generic_dir: The path to the generic directory. The content of this folder is copied to
                each client folder. If the file in the generic folder already exists in the
                client folder, it is not copied. Folders in the generic folder are not copied.
                The first clientfolder is used as the coordinator.
        """
        # checks
        if len(clientfolders) < 2:
            raise ValueError("At least two client folders are required")
        if len(clientfolders) != len(outputfolders):
            raise ValueError("The number of client folders and output folders must be the same")
        # basic variables
        self.num_clients = len(clientfolders)
        self.shared_dict = FedLearnSimulationGateway(num_clients=self.num_clients)
        self.created_files = []

        # copy files from the generic folder to each client folder
        for clientfolder in clientfolders:
            # copy files from the generic folder to the client folder
            if generic_dir is not None:
                for root, _, files in os.walk(generic_dir):
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(clientfolder, file)
                        if os.path.exists(dst_file):
                            if os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                                print("WARNING: File from generic folder already exists in client " +\
                                    "but generic file is newer. Overwriting")
                                shutil.copy(src_file, dst_file)
                                self.created_files.append(dst_file)
                            else:
                                print("WARNING: File from generic folder already exists in client " +\
                                    "and is newer. Skipping copy of the file from generic folder")
                        else:
                            # file doesn't exist yet, copy it
                            shutil.copy(src_file, dst_file)
                            self.created_files.append(dst_file)

        # create the client instances
        self.clients: List[LocalFedLearningSimulator] = []
        for i, clientfolder in enumerate(clientfolders):
            is_coordinator = i == 0
            client = LocalFedLearningSimulator(is_coordinator=is_coordinator,
                                               client_id=i,
                                               num_clients=self.num_clients,
                                               inputfolder=clientfolder,
                                               outputfolder=outputfolders[i],
                                               gateway=self.shared_dict)
            self.clients.append(client)

    def cleanup_created_files(self):
        """
        Removes any files that were added to client folders during the init process
        from the generic folder.
        """
        for file in self.created_files:
            os.remove(file)
        self.created_files = []
