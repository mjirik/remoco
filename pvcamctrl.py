#  /usr/bin/python
# -*- coding: utf-8 -*-
import socket, traceback
# use android app sense to transmit
import xml.etree.ElementTree as etree

import time
import numpy as np
import os.path as op
import threading
import math

# udp  pakety pomocí aplikace Sense Free

# TODO keyboard interrupt works when there is no timeout. The problem is in socket blocking mode. Fix this.

try:
    # from paraview.simple import *
    import paraview.simple as pasi
    paraview_loaded = True
except:
    print("paraview package not loaded")
    paraview_loaded = False

# else:


def root_from_xmlstr(xmlstr):
    # print("=== xmlstr ===")
    # print(xmlstr)
    fixed_xmlstr = fix_xml(xmlstr)
    root = etree.fromstring(fixed_xmlstr)
    return root


def fix_xml(xmlstr):
    """
    Fix broken XML from Android app Sense. The root element is added.
    :return: fixed xmlstring
    """

    i = xmlstr.find("?>")
    fixed_xmlstring = xmlstr[:i + 2] + "<root>" + xmlstr[i + 2:] + "</root>"
    # print(fixed_xmlstring)
    return fixed_xmlstring

def get_rotation_vector(root):
    rotation = root.find("RotationVector")
    return [
        float(rotation.find("RotationVector1").text),
        float(rotation.find("RotationVector2").text),
        float(rotation.find("RotationVector3").text),
    ]


class StreamReader(threading.Thread):
    # class StreamReader():

    def __init__(self, show_debug=False, maximum_timeouts=10):
        threading.Thread.__init__(self)
        self.show_debug = show_debug
        host = ''
        port = 50000
        port = 5555
        # port = 22
        import socket

        print((([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [
            [(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in
             [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0])
        print("port ", port)

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(2)
        s.bind((host, port))
        self.socket = s
        self.filename = op.expanduser("~/camerastream.npy")
        self.maximum_timeouts = maximum_timeouts



    def iteration(self):
        # message, address = s.recvfrom(8192)
        message, address = self.socket.recvfrom(1024)
        # print(message)
        # print("recived from " + str(address))
        messageString = message.decode("utf-8")
        # print(messageString)
        root = root_from_xmlstr(messageString)
        rotation = get_rotation_vector(root)
        rotation_str = ['{:.2f}'.format(i) for i in rotation]
        eangles = rvector_to_eangles(rotation)
        eangles_str = ['{:.2f}'.format(i) for i in eangles]
        print(rotation_str, eangles_str, "recived from " + str(address) )
        # np.asarray(rotation)
        np.save(self.filename, rotation)
        return rotation

    def save_random(self):
        """
        Save to output file random values. Used for debugging.
        :return:
        """
        random_rotation = np.random.rand(3)
        np.save(self.filename, random_rotation)
        return random_rotation

    def run(self):
        self.iterations()

    def iterations(self):
        print("iterations started")
        show = self.show_debug
        rotations = []
        # rotations_euler = []
        timeouts_number = 0
        stay_in_loop = True
        try:
            while stay_in_loop:
                # sleep(0.5)
                # print("yaooo")
                try:
                    rotation = self.iteration()
                    if show:
                        rotations.append(rotation)
                        # rotations_euler.append(rvector_to_eangles(rotations))
                # print(messageString)
                except socket.timeout as e:
                    err = e.args[0]
                    # this next if/else is a bit redundant, but illustrates how the
                    # timeout exception is setup
                    print(e)
                    if err == 'timed out':
                        time.sleep(1)
                        # try:
                        #     pass
                        #     # time.sleep(1)
                        # except KeyboardInterrupt as ke:
                        #     print(ke)
                        #     stay_in_loop = False
                        random_rotation = self.save_random()
                        print('recv timed out, retry later, generated random rotation: ', random_rotation)

                        timeouts_number += 1
                        if timeouts_number >= self.maximum_timeouts:
                            break

                        continue
                    else:
                        print(e)
                except KeyboardInterrupt as ke:
                    print("Stream reader interrupted by keyboard")
                    self.socket.close()
                    # stay_in_loop = False
                    break

                except ValueError as ve:

                    print(ve)
                    traceback.print_exc()
                    print("Numeric error in angles conversion. It is not a real problem.")

                # except Exception as exc:
                #     print(str(exc))
                #     print('Server closing')
                #     self.socket.close()
                #     break

        except KeyboardInterrupt as ke:
            print("Stream reader interrupted by keyboard")
            stay_in_loop = False
            self.socket.close()
            # break
            # raise ke
                    # sys.exit(1)
        if show:
            print("show")
            import matplotlib.pyplot as plt
            rotations = np.asarray(rotations)
            x = range(len(rotations))
            plt.plot(
                x, rotations[:, 0],
                x, rotations[:, 1],
                x, rotations[:, 2],
            )
            plt.legend(["0","1","2"])
            plt.show()
            # rotations_euler = np.asarray(rotations_euler)
            # x = range(len(rotations))
            # plt.plot(
            #     x, rotations_euler[:, 0],
            #     x, rotations_euler[:, 1],
            #     x, rotations_euler[:, 2],
            # )
            # plt.legend(["0","1","2"])
            # plt.show()


    #used for debugging

# class CameraMover(threading.Thread):
class CameraMover():
    def __init__(
            self,
            hostname=None,
            stream_fn="~/camerastream.npy",
            timestep=0.2,
            state_fn=None,
            view_size=None,
            measurement_compensation_factor=1.25,
            debug_rotation=False

    ):
        """

        :param hostname:
        :param stream_fn: .npy file with saved numpy array with rotation vector
        :param timestep: Refresh time
        :param state_fn: .pvsm file with saved paraview state
        """
        # threading.Thread.__init__(self)
        self.prev_angle_deg = None
        self.stream_fn = op.expanduser(stream_fn)
        # hostname = "localhost"
        self.view_size = view_size
        self.hostname = hostname
        self.debug_rotation = debug_rotation
        self.measurement_compensation_factor = measurement_compensation_factor
        if self.hostname is not None:
            self.connect = pasi.Connect(self.hostname)

        print("Success binding")
        self.timestep = timestep
        if state_fn is not None:
            self.load_state(state_fn)
        self.init_view()



    def rvector_to_angle_deg0(self, rotation):
        # print(rotation)
        # this is experiment based compensation
        rt0 = rotation[2] * 180
        rt = rt0 * self.measurement_compensation_factor
        #limit
        rt = np.max([np.min([rt, 180.0]), -180.0])
        # print(rt)
        # angle_deg = np.rad2deg(np.arcsin(rt)) * 2
        angle_deg = rt
        if self.debug_rotation:
            print("rotation ", rt0, angle_deg)
        return angle_deg
        # return rotation[2] * 180

    def rvector_to_angle_deg(self, rotation):
        eangles = rvector_to_eangles(rotation)
        return eangles[2]

    def init_view(self):
        pasi.SetActiveView(pasi.GetRenderView())
        if paraview_loaded:
            camera = pasi.GetActiveCamera()
            self.camera = camera
            # cm = CameraMover(camera)
            print("Camera loaded")

            if self.view_size is not None:
                pasi.GetActiveView().ViewSize = self.view_size

    def load_state(self, state_filename):
        pasi.servermanager.LoadState(state_filename)

    def camera_rotate(self, angle_deg):
        if self.prev_angle_deg is not None:
            self.camera.Azimuth(self.prev_angle_deg - angle_deg)
        self.prev_angle_deg = angle_deg
        pasi.Render()

    def iteration(self):
        try:
            rotation = np.load(self.stream_fn)
            if paraview_loaded:

                self.camera_rotate(self.rvector_to_angle_deg(rotation))
                # self.camera_rotate(self.rvector_to_angle_deg0(rotation))

                # self.camera_rotate(rotation[2] * 180)
                # self.rvector_to_angle_deg(rotation=rotation)
        except Exception as e:
            print(e)

    def run(self):
        self.iterations(n=300)

    def iterations(self, n=10):
        if n is None:
            while 1:
                self.iteration()

        else:
            for i in range(n):
                print("iteration {}".format(i))
                self.iteration()
                time.sleep(self.timestep)

def quaternionvec_from_rotationvec(rotation):
    """
    Calculate 4th part of normalized quaternion
    :param rotation:
    :return:
    """
    # Get full normalized quaternion
    qx = float(rotation[0])
    qy = float(rotation[1])
    qz = float(rotation[2])
    qw = (1 - qx**2 - qy**2 - qz**2)**0.5
    return qw, qx, qy, qz

def rvector_to_eangles(rotation):
    # TODO finish


    import math
    # Get Euler angles ex, ey, ez
    # ex = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    # ey = np.arcsin(2 * (qw * qy - qz * qx))
    # ey = math.asin(2 * (qw * qy - qz * qx))
    # ez = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz**2))
    # eangles1 = np.array([ex, ey, ez])

    # from . import transformations
    # import transformations
    # eangles2 = transformations.euler_from_quaternion([qw, qx, qy, qz])
    qw, qx, qy, qz = quaternionvec_from_rotationvec(rotation)
    ex, ey, ez = quaternion_to_euler_angle(qw, qx, qy, qz)
    eangles3 = np.array([ex, ey, ez])
    return eangles3



def quaternion_to_euler_angle(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='CameraControler'
    )

    # parser.add_argument(
    #     '-i', '--inputfile',
    #     default=None,
    #     help='Input file/directory. Generates sample data, if not set.')
    parser.add_argument(
        '-sr', "--stream-reader",
        action='store_true',
        help='Read and parse data from stream')
    parser.add_argument(
        '-srs', "--stream-reader-show",
        action='store_true',
        help='Record data from stream reader and show the plot')

    parser.add_argument(
        '-pv', "--paraview-show",
        action='store_true',
        help='Turn on paraview visualization')
    parser.add_argument(
        '-pvf', '--paraview-file',
        default=None,
        help='Input paraview state file')
    parser.add_argument(
        '-pvws', '--paraview-window-size',
        default=None,
        type=int,
        nargs=2,
        help='Size of the paraview visualization window'
    )
    parser.add_argument(
        '-pvcf', '--paraview-compensation-factor',
        type=float,
        default=1.25,
        help='Visualization compensation factor. 1.0 means no compensation to rotation_vector[2]')
    parser.add_argument(
        '-pvdr', "--paraview-debug-rotation",
        action='store_true',
        help='Turn on paraview rotation debug prints')

    args = parser.parse_args()

    if not args.stream_reader and not args.paraview_show:
        args.stream_reader = True

    if args.stream_reader:
        cm = StreamReader(show_debug=args.stream_reader_show, maximum_timeouts=10)
        # cm.iterations() #  not start new thread
        cm.start() #  start new thread

    if args.paraview_show:
        cm = CameraMover(
            state_fn=args.paraview_file,
            view_size=args.paraview_window_size,
            debug_rotation=args.paraview_debug_rotation,
            measurement_compensation_factor=args.paraview_compensation_factor
        )
        cm.iterations(None)

# Example of XML data received:
# <Node Id>node12</Node Id>
# <GPS>
# <Latitude>1.123123</Latitude>
# <Longitude>234.1231231</Longitude>
# <Accuracy>40.0</Accuracy>
# </GPS>
# <Accelerometer>
# <Accelerometer1>0.38444442222</Accelerometer1>
# <Accelerometer2>0.03799999939</Accelerometer2>
# <Accelerometer3>9.19400000331</Accelerometer3>
# </Accelerometer>
# <TimeStamp>1370354489083</TimeStamp>
if __name__ == "__main__":
    main()
