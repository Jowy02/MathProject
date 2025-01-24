import customtkinter
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import numpy as np

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class Arcball(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        # Orientation vars. Initialized to represent 0 rotation
        self.quat = np.array([[1],[0],[0],[0]])
        self.rotM = np.eye(3)
        self.AA = {"axis": np.array([[0],[0],[0]]), "angle":0.0}
        self.rotv = np.array([[0],[0],[0]])
        self.euler = np.array([[0],[0],[0]])
        self.prevQuat = np.array([[1],[0],[0],[0]])
        self.prevPoint = np.array([0,0,0])
        self.rot = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        # configure window
        self.title("Holroyd's arcball")
        self.geometry(f"{1100}x{580}")
        self.resizable(False, False)
        self.prueba = False

        self.grid_columnconfigure((0,1), weight=0   )
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Cube plot
        self.init_cube()

        self.canvas = FigureCanvasTkAgg(self.fig, self)  # A tk.DrawingArea.
        self.bm = BlitManager(self.canvas,[self.facesObj])
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.pressed = False #Bool to bypass the information that mouse is clicked
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        
        # Reset button
        self.resetbutton = customtkinter.CTkButton(self, text="Reset", command=self.resetbutton_pressed)
        self.resetbutton.grid(row=3, column=0, padx=(0, 0), pady=(5, 20), sticky="ns")
        
        # Selectable atti
        self.tabview = customtkinter.CTkTabview(self, width=150, height=150)
        self.tabview.grid(row=0, column=1, padx=(0, 20), pady=(20, 0), sticky="nsew")
        self.tabview.add("Axis angle")
        self.tabview.add("Rotation vector")
        self.tabview.add("Euler angles")
        self.tabview.add("Quaternion")

        # Selectable atti: AA
        self.tabview.tab("Axis angle").grid_columnconfigure(0, weight=0)  # configure grid of individual tabs
        self.tabview.tab("Axis angle").grid_columnconfigure(1, weight=0)  # configure grid of individual tabs

        self.label_AA_axis= customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Axis:")
        self.label_AA_axis.grid(row=0, column=0, rowspan=3, padx=(80,0), pady=(45,0), sticky="e")

        self.entry_AA_ax1 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax1.insert(0,"1.0")
        self.entry_AA_ax1.grid(row=0, column=1, padx=(5, 0), pady=(50, 0), sticky="ew")

        self.entry_AA_ax2 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax2.insert(0,"0.0")
        self.entry_AA_ax2.grid(row=1, column=1, padx=(5, 0), pady=(5, 0), sticky="ew")

        self.entry_AA_ax3 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax3.insert(0,"0.0")
        self.entry_AA_ax3.grid(row=2, column=1, padx=(5, 0), pady=(5, 10), sticky="ew")

        self.label_AA_angle = customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Angle (degrees):")
        self.label_AA_angle.grid(row=3, column=0, padx=(120,0), pady=(10, 20),sticky="w")
        self.entry_AA_angle = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_angle.insert(0,"0.0")
        self.entry_AA_angle.grid(row=3, column=1, padx=(5, 0), pady=(0, 10), sticky="ew")

        self.button_AA = customtkinter.CTkButton(self.tabview.tab("Axis angle"), text="Apply", command=self.apply_AA, width=180)
        self.button_AA.grid(row=5, column=0, columnspan=2, padx=(0, 0), pady=(5, 0), sticky="e")

        # Selectable atti: rotV
        self.tabview.tab("Rotation vector").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Rotation vector").grid_columnconfigure(1, weight=0)
        
        self.label_rotV= customtkinter.CTkLabel(self.tabview.tab("Rotation vector"), text="rot. Vector:")
        self.label_rotV.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(45,0), sticky="e")

        self.entry_rotV_1 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_1.insert(0,"0.0")
        self.entry_rotV_1.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_rotV_2 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_2.insert(0,"0.0")
        self.entry_rotV_2.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_rotV_3 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_3.insert(0,"0.0")
        self.entry_rotV_3.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_rotV = customtkinter.CTkButton(self.tabview.tab("Rotation vector"), text="Apply", command=self.apply_rotV, width=180)
        self.button_rotV.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Euler angles
        self.tabview.tab("Euler angles").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Euler angles").grid_columnconfigure(1, weight=0)
        
        self.label_EA_roll= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="roll (degrees):")
        self.label_EA_roll.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_EA_pitch= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="pitch (degrees):")
        self.label_EA_pitch.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_EA_yaw= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="yaw (degrees):")
        self.label_EA_yaw.grid(row=2, column=0, rowspan=3, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_EA_roll = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_roll.insert(0,"0.0")
        self.entry_EA_roll.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_EA_pitch = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_pitch.insert(0,"0.0")
        self.entry_EA_pitch.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_EA_yaw = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_yaw.insert(0,"0.0")
        self.entry_EA_yaw.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_EA = customtkinter.CTkButton(self.tabview.tab("Euler angles"), text="Apply", command=self.apply_EA, width=180)
        self.button_EA.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Quaternion
        self.tabview.tab("Quaternion").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Quaternion").grid_columnconfigure(1, weight=0)
        
        self.label_quat_0= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q0:")
        self.label_quat_0.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_quat_1= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q1:")
        self.label_quat_1.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_2= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q2:")
        self.label_quat_2.grid(row=2, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_3= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q3:")
        self.label_quat_3.grid(row=3, column=0, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_quat_0 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_0.insert(0,"1.0")
        self.entry_quat_0.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_quat_1 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_1.insert(0,"0.0")
        self.entry_quat_1.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_2 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_2.insert(0,"0.0")
        self.entry_quat_2.grid(row=2, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_3 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_3.insert(0,"0.0")
        self.entry_quat_3.grid(row=3, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_quat = customtkinter.CTkButton(self.tabview.tab("Quaternion"), text="Apply", command=self.apply_quat, width=180)
        self.button_quat.grid(row=4, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Rotation matrix info
        self.RotMFrame = customtkinter.CTkFrame(self, width=150)
        self.RotMFrame.grid(row=1, column=1, rowspan=3, padx=(0, 20), pady=(20, 20), sticky="nsew")

        self.RotMFrame.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.label_RotM= customtkinter.CTkLabel(self.RotMFrame, text="RotM = ")
        self.label_RotM.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(20,0), sticky="e")

        self.entry_RotM_11= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_11.insert(0,"1.0")
        self.entry_RotM_11.configure(state="disabled")
        self.entry_RotM_11.grid(row=0, column=1, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_12= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_12.insert(0,"0.0")
        self.entry_RotM_12.configure(state="disabled")
        self.entry_RotM_12.grid(row=0, column=2, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_13= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_13.insert(0,"0.0")
        self.entry_RotM_13.configure(state="disabled")
        self.entry_RotM_13.grid(row=0, column=3, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_21= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_21.insert(0,"0.0")
        self.entry_RotM_21.configure(state="disabled")
        self.entry_RotM_21.grid(row=1, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_22= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_22.insert(0,"1.0")
        self.entry_RotM_22.configure(state="disabled")
        self.entry_RotM_22.grid(row=1, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_23= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_23.insert(0,"0.0")
        self.entry_RotM_23.configure(state="disabled")
        self.entry_RotM_23.grid(row=1, column=3, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_31= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_31.insert(0,"0.0")
        self.entry_RotM_31.configure(state="disabled")
        self.entry_RotM_31.grid(row=2, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_32= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_32.insert(0,"0.0")
        self.entry_RotM_32.configure(state="disabled")
        self.entry_RotM_32.grid(row=2, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_33= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_33.insert(0,"1.0")
        self.entry_RotM_33.configure(state="disabled")
        self.entry_RotM_33.grid(row=2, column=3, padx=(2,0), pady=(2,0), sticky="ew")
    

    def update_entries(self, quaternion, axis, angle, rotV, roll, pitch, yaw):
        # Update screen
        # Quaternion
        self.entry_quat_0.delete(0, "end")
        self.entry_quat_0.insert(0, str(quaternion[0]))
        self.entry_quat_1.delete(0, "end")
        self.entry_quat_1.insert(0, str(quaternion[1]))
        self.entry_quat_2.delete(0, "end")
        self.entry_quat_2.insert(0, str(quaternion[2]))
        self.entry_quat_3.delete(0, "end")
        self.entry_quat_3.insert(0, str(quaternion[3]))

        # Axis-Angle
        self.entry_AA_ax1.delete(0, "end")
        self.entry_AA_ax1.insert(0, str(axis[0]))
        self.entry_AA_ax2.delete(0, "end")
        self.entry_AA_ax2.insert(0, str(axis[1]))
        self.entry_AA_ax3.delete(0, "end")
        self.entry_AA_ax3.insert(0, str(axis[2]))
        self.entry_AA_angle.delete(0, "end")
        self.entry_AA_angle.insert(0, str(np.degrees(angle)))  # Convertir ángulo a grados

        # Rotation Vector
        self.entry_rotV_1.delete(0, "end")
        self.entry_rotV_1.insert(0, str(rotV[0]))
        self.entry_rotV_2.delete(0, "end")
        self.entry_rotV_2.insert(0, str(rotV[1]))
        self.entry_rotV_3.delete(0, "end")
        self.entry_rotV_3.insert(0, str(rotV[2]))

        # Euler Angles
        self.entry_EA_roll.delete(0, "end")
        self.entry_EA_roll.insert(0, str(roll))  # Convertir a grados
        self.entry_EA_pitch.delete(0, "end")
        self.entry_EA_pitch.insert(0, str(pitch))  # Convertir a grados
        self.entry_EA_yaw.delete(0, "end")
        self.entry_EA_yaw.insert(0, str(yaw))  # Convertir a grados

    def resetbutton_pressed(self):
        """
        Event triggered function on the event of a push on the button Reset
        """
        
        self.M = np.array(
            [[ -1,  -1, 1],   # Node 0
            [ -1,   1, 1],    # Node 1
            [1,   1, 1],      # Node 2
            [1,  -1, 1],      # Node 3
            [-1,  -1, -1],    # Node 4
            [-1,  1, -1],     # Node 5
            [1,   1, -1],     # Node 6
            [1,  -1, -1]], dtype=float).transpose() # Node 7

        self.update_cube() # Update the cube

        R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) # Update rotation. Initial rotation
        self.rot = R
        self.prevQuat = np.array([[1],[0],[0],[0]])
        self.updateText(self.rot,False)

        # Update information screen
        quat = np.array([1.0,0.0,0.0,0.0])
        axis = np.array([1.0,0.0,0.0])
        angle=0.0
        rotV=axis*angle
        roll=0.0
        pitch=0.0
        yaw=0.0
        
        self.update_entries(quat, axis, angle, rotV, roll, pitch, yaw)
       

        pass


    def apply_AA(self):
        """
        Event triggered function on the event of a push on the button button_AA
        """
        # Example on hot to get values from entries:
        angle = (float(self.entry_AA_angle.get()))
        angle = angle * np.pi / 180
        
        # Get the values of the components of the rotation axis
        axis=np.array([float(self.entry_AA_ax1.get()),float(self.entry_AA_ax2.get()), float(self.entry_AA_ax3.get())])
        
        # Normalize axis
        axis_normalize = axis/np.linalg.norm(axis)

        # Convertir a matriz de rotación y quaternion
        self.M, new_quaternion=self.axisAngle_to_quaternion(axis_normalize,angle)
    
        # Update Rotation
        self.rot=self.M
        self.update_cube()
       
        self.fig.canvas.draw_idle()
        self.rot = self.quaternion_to_rotation_matrix(self.prevQuat)
        self.updateText(self.rot,False)

        # Calculate updated values
        # Calculate rotation vector
        rotV = angle*axis_normalize
       
        # Calculate Euler angles
        yaw,pitch,roll=self.rotation_matrix_to_euler_angles( self.rot )

        self.update_entries(new_quaternion, axis, angle, rotV, roll, pitch, yaw)

    
        pass

    
    def apply_rotV(self):
        """
        Event triggered function on the event of a push on the button button_rotV 
        """
        # Rotation vector
        r=np.array([float(self.entry_rotV_1.get()),float(self.entry_rotV_2.get()), float(self.entry_rotV_3.get())])

        # Normalize vector
        vector = r/np.linalg.norm(r)
        angle=np.linalg.norm(vector)

        self.M,new_quaternion=self.axisAngle_to_quaternion(vector,angle)
     
        # Update Rotation
        self.rot=self.M
        self.update_cube()
       
        self.fig.canvas.draw_idle()
        self.rot = self.quaternion_to_rotation_matrix(self.prevQuat)
        self.updateText(self.rot,False)

        # Calculate updated values
        # Calculate Euler angles
        yaw,pitch,roll=self.rotation_matrix_to_euler_angles( self.rot )

        self.update_entries(new_quaternion, vector, angle, r, roll, pitch, yaw)


        pass

    
    def apply_EA(self):
        """
        Event triggered function on the event of a push on the button button_EA
        """
        # Get the values
        roll= float(self.entry_EA_roll.get())
        pitch=float(self.entry_EA_pitch.get())
        yaw= float(self.entry_EA_yaw.get())

        # Convert angle from degrees
        roll=roll*np.pi/180
        pitch=pitch*np.pi/180
        yaw=yaw*np.pi/180

        # Define the rotation matrix for yaw 
        Rz=np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]])
      
        # Define the rotation matrix for pitch 
        Ry=np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]])
        
        # Define the rotation matrix for roll 
        Rx=np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]])
        
        # Rotation matrix
        R=Rz.dot(Ry).dot(Rx)
        
        # Convert the rotation matrix to a quaternion   
        axis,angle=self.rotationMatrix_to_axisAngle(R)
        self.M,new_quaternion=self.axisAngle_to_quaternion(axis,angle)
      
        # Update Rotation
        self.update_cube()
        self.fig.canvas.draw_idle()

        self.rot = self.quaternion_to_rotation_matrix(self.prevQuat)
        self.updateText(self.rot,False)

        # Calculate rotation vector
        rotV=np.array([angle*axis[0],angle*axis[1],angle*axis[2] ])
        
        self.update_entries(new_quaternion, axis, angle, rotV, roll, pitch, yaw)

        pass

    
    def apply_quat(self):
        """
        Event triggered function on the event of a push on the button button_quat
        """
        # Get the quaternion values
        q0 = float(self.entry_quat_0.get())
        q1 = float(self.entry_quat_1.get())
        q2 = float(self.entry_quat_2.get())
        q3 = float(self.entry_quat_3.get())

        # The quaternion entered by the user
        enter_quat = np.array([q0, q1, q2, q3])
        user_quat=enter_quat/np.linalg.norm(enter_quat)
      
        # Combine the new quaternion with the previous quaternion
        self.prevQuat =  user_quat

        # Rotate each column vector of the vertices 
        q = np.squeeze(self.prevQuat)
        rotated = np.array([
            self.quaternion_rotate_vector(q, self.M[:, i])
            for i in range(self.M.shape[1])])
        self.M=rotated.T

        self.update_cube() # Update the cube
        self.fig.canvas.draw_idle()
        
        self.prevPoint = np.array([q1, q2, q3])
        
        self.rot = self.quaternion_to_rotation_matrix(self.prevQuat)
        self.updateText(self.rot,False)
        
        # Pass through the matrix rotation, obtain the angle and the vector.
        axis,angle=self.rotationMatrix_to_axisAngle(self.rot)

        # Calculate Euler angles
        yaw,pitch,roll=self.rotation_matrix_to_euler_angles( self.rot )

        # Calculate rotation vector
        rotV=np.array([angle*axis[0],angle*axis[1],angle*axis[2] ])

        self.update_entries(enter_quat, axis, angle, rotV, roll, pitch, yaw)

        pass

    def rotationMatrix_to_axisAngle(self,R):
        if not np.allclose(R @ np.transpose(R), np.eye(3)): return None

        trace = np.trace(R)
        angle = np.arccos((trace - 1) / 2)
        axis = np.zeros(3)

        if angle == 0:
            # Handle special case when sin(theta) is 0 (e.g., identity matrix)
            axis = np.array([1, 0, 0]) #Eje arbitrario
            return axis, 0

        if np.isclose(angle, np.pi):
        # Determine what is the maximum element on the diagonal
            max_index = R[0, 0]
            if max_index < R[1, 1] :
                max_index = R[1, 1]

            if max_index < R[2, 2] :
                max_index = R[2, 2]

            # Depending on which element is the largest, calculate the components of the axis of rotation
            if max_index == R[0, 0] : # Element [0, 0] is the largest
                axis[0] = np.sqrt((R[0, 0] + 1) / 2) # Positive principal component
                axis[1] = R[0, 1] / (2 * axis[0])
                axis[2] = R[0, 2] / (2 * axis[0])

            if max_index == R[1, 1] :  # Element [1, 1] is the largest
                axis[1] = np.sqrt((R[1, 1] + 1) / 2) # Positive principal component
                axis[0] = R[1, 0] / (2 * axis[1])
                axis[2] = R[1, 2] / (2 * axis[1])

            if max_index == R[2, 2] :   # Element [2, 2] is the largest
                axis[2] = np.sqrt((R[2, 2] + 1) / 2) # Positive principal component
                axis[0] = R[2, 0] / (2 * axis[2])
                axis[1] = R[2, 1] / (2 * axis[2])
            return axis, angle

        else:
            axis[0] = (R[2, 1] - R[1, 2]) / (2 * np.sin(angle))
            axis[1] = (R[0, 2] - R[2, 0]) / (2 * np.sin(angle))
            axis[2] = (R[1, 0] - R[0, 1]) / (2 * np.sin(angle))

        return axis, angle
    
    def axisAngle_to_quaternion(self,axis,angle):
        # Calculate the magnitude (norm) of the rotation axis vector
        axis = axis / np.linalg.norm(axis)

        # Quaternion formula:
        new_q=np.array([[np.cos(angle/2)],
                        [np.sin(angle/2)*axis[0]],
                        [np.sin(angle/2)*axis[1]],
                        [np.sin(angle/2)*axis[2]]])
        
        # Combine the new quaternion with the previous quaternion
        self.prevQuat = new_q
        # Rotate each column vector of the vertices 
        q = np.squeeze(self.prevQuat)
        rotated = np.array([
            self.quaternion_rotate_vector(q, self.M[:, i])
            for i in range(self.M.shape[1])])
        return rotated.T, q

    def rotation_matrix_to_euler_angles(self,R):
        if R[2, 0]!=1 and R[2, 0]!=-1:
            pitch = np.arcsin(-R[2, 0]) # theta
            cos=1/np.cos(pitch)
            roll=np.arctan2(R[2,1]*cos,R[2,2]*cos) # phi
            yaw=np.arctan2(R[1,0]*cos,R[0,0]*cos) # psi

        if R[2, 0]==1:
            pitch=-(np.pi/2) # -90 degrees # theta
            roll=np.arctan2(-R[0,1],-R[0,2]) # phi
            yaw=0 # 0 degrees #psi

        if R[2, 0]==-1:
            pitch=np.pi/2 # 90 degrees #theta
            roll=np.arctan2(R[0,1],R[0,2]) # phi
            yaw=0 # 0 degrees #psi
        
        # Change to degrees
        yaw=yaw*180/np.pi
        pitch=pitch*180/np.pi
        roll=roll*180/np.pi
        return (yaw, pitch, roll)

    def onclick(self, event):
        """
        Event triggered function on the event of a mouse click inside the figure canvas
        """
        if event.button:
            self.pressed = True # Bool to control(activate) a drag (click+move)
            x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) # Extract viewport coordinates
           
            self.prevPoint = self.takePoint(x_fig,y_fig)
            self.prevPoint /= np.linalg.norm(self.prevPoint)

          
    def onmove(self, event):
        """
        Event triggered function on the event of a mouse motion
        """
        # Example
        if self.pressed:  # Only triggered if previous click
            x_fig, y_fig = self.canvas_coordinates_to_figure_coordinates(event.x, event.y)  # Extract viewport coordinates
        
            m1 = self.takePoint(x_fig, y_fig)
            m0 = self.prevPoint

            # Normalize Vectors
            m1 /= np.linalg.norm(m1)
            norm_m1 = np.linalg.norm(m1)
            norm_m0 = np.linalg.norm(m0)
     
            # Dot product and cross product
            dot_product = np.dot(m0, m1)
            cross_product = np.cross(m0, m1)

            # Calculate the delta quaternion
            self.prevQuat = np.array([(norm_m0 * norm_m1) + dot_product,*cross_product])

            # Normalize the delta quaternion
            self.prevQuat /= np.linalg.norm(self.prevQuat)

            # Rotate the matrix based on the quaternion
            q = np.squeeze(self.prevQuat)
            rotated = np.array([
                self.quaternion_rotate_vector(q, self.M[:, i])
                for i in range(self.M.shape[1])
            ])
            
            self.M = rotated.T
            self.update_cube()  # Update the cube
            self.updateText(self.rot, True)
            self.prevPoint = m1

    def onrelease(self, event):
        """
        Event triggered function on the event of a mouse release
        """
        self.rot = self.quaternion_to_rotation_matrix(self.prevQuat)
        self.prueba = False
        self.pressed = False  # Bool to control(deactivate) a drag (click+move)

    def quaternion_rotate_vector(self, q, v):
        """Rotate a vector v using the quaternion q."""
        w, x, y, z = q
        q_vector = np.array([0, *v])
        q_conjugate = np.array([w, -x, -y, -z])

        result = self.quaternion_multiply(self.quaternion_multiply(q, q_vector), q_conjugate)
        return result[1:]  # Return only the vector part

    def takePoint(self, x_fig, y_fig):
        """Generate a 3D vector"""
        r = np.sqrt(3)
        distance = (x_fig ** 2) + (y_fig ** 2)

        if distance < (r ** 2) / 2:
            z_fig = np.sqrt((r ** 2) - (x_fig ** 2) - (y_fig ** 2))
            m = np.array([x_fig, y_fig, z_fig])

        elif distance >= (r ** 2) / 2:
            z_fig = (r ** 2) / (2 * np.sqrt(distance))
            m = np.array([x_fig, y_fig, z_fig])
            div = np.linalg.norm(m)
            m = (r * m) / div

        return m

    def quaternion_multiply(self, q, p):
        """Multiply two quaternions q1 and q2."""
        p = np.squeeze(p)
        
        q0, qv = q[0], q[1:]
        p0, pv = p[0], p[1:]
        qp = np.zeros(4)
        qp[0] = q0 * p0 - np.dot(qv, pv)
        qp[1:] = q0 * pv + p0 * qv + np.cross(qv, pv)

        return qp

    def quaternion_to_rotation_matrix(self, q):

        w, x, y, z = q
        # Calcula los elementos de la matriz de rotación
        R = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x**2 + y**2)]])

        # Build the rotation matrix using the expanded formula
        R = R.reshape(3, -1)
        R[np.isclose(R,0)] = 0

        return R

    def updateText(self, R, full):
        entries = [
            [self.entry_RotM_11, self.entry_RotM_12, self.entry_RotM_13],
            [self.entry_RotM_21, self.entry_RotM_22, self.entry_RotM_23],
            [self.entry_RotM_31, self.entry_RotM_32, self.entry_RotM_33],
        ]

        # Iterar sobre filas y columnas
        for i in range(3):
            for j in range(3):
                entry = entries[i][j]
                entry.configure(state="normal")
                entry.delete(0, "end")
                entry.insert(0, R[i, j])
                entry.configure(state="disabled")

                self.entry_quat_0.delete(0, "end")

        if (full):
            self.rot = self.quaternion_to_rotation_matrix(self.prevQuat)
            
            # Pass through the matrix rotation, obtain the angle and the vector.
            axis,angle=self.rotationMatrix_to_axisAngle(self.rot)

            # Calculate Euler angles
            yaw,pitch,roll=self.rotation_matrix_to_euler_angles( self.rot )

            # Calculate rotation vector
            rotV=np.array([angle*axis[0],angle*axis[1],angle*axis[2] ])

            self.entry_quat_0.insert(0, str(self.prevQuat[0]))  # Initial value for q0
            self.entry_quat_1.delete(0, "end")
            self.entry_quat_1.insert(0, str(self.prevQuat[1]))  # Initial value for q1
            self.entry_quat_2.delete(0, "end")
            self.entry_quat_2.insert(0, str(self.prevQuat[2]))  # Initial value for q2
            self.entry_quat_3.delete(0, "end")
            self.entry_quat_3.insert(0, str(self.prevQuat[3]))  # Initial value for q3


            ##Update data apply_rotV        
            self.entry_AA_ax1.delete(0, "end")
            self.entry_AA_ax1.insert(0, str(axis[0]))  # Initial value for axis1
            self.entry_AA_ax2.delete(0, "end")
            self.entry_AA_ax2.insert(0, str(axis[1]))  # Initial value for axis2
            self.entry_AA_ax3.delete(0, "end")
            self.entry_AA_ax3.insert(0, str(axis[2]))  # Initial value for axis3
            self.entry_AA_angle.delete(0, "end")
            self.entry_AA_angle.insert(0, str(angle))  # Initial value for angle

            ##Update data apply_rotV        
            self.entry_rotV_1.delete(0, "end")
            self.entry_rotV_1.insert(0, str(rotV[0]))  # Initial value for rotV1
            self.entry_rotV_2.delete(0, "end")
            self.entry_rotV_2.insert(0, str(rotV[1]))  # Initial value for rotV2
            self.entry_rotV_3.delete(0, "end")
            self.entry_rotV_3.insert(0, str(rotV[2]))  # Initial value for rotV3

            ##Update data apply_EA
            self.entry_EA_roll.delete(0, "end")
            self.entry_EA_roll.insert(0, str(roll))  # Initial value for roll
            self.entry_EA_pitch.delete(0, "end")
            self.entry_EA_pitch.insert(0, str(pitch))  # Initial value for pitch
            self.entry_EA_yaw.delete(0, "end")
            self.entry_EA_yaw.insert(0, str(yaw))  # Initial value for yaw

    def init_cube(self):
        """
        Initialization function that sets up cube's geometry and plot information
        """
        self.M = np.array(
            [[ -1,  -1, 1],   # Node 0
            [ -1,   1, 1],    # Node 1
            [1,   1, 1],      # Node 2
            [1,  -1, 1],      # Node 3
            [-1,  -1, -1],    # Node 4
            [-1,  1, -1],     # Node 5
            [1,   1, -1],     # Node 6
            [1,  -1, -1]], dtype=float).transpose() # Node 7

        self.con = [
            [0, 1, 2, 3], # Face 1
            [4, 5, 6, 7], # Face 2
            [3, 2, 6, 7], # Face 3
            [0, 1, 5, 4], # Face 4
            [0, 3, 7, 4], # Face 5
            [1, 2, 6, 5]] # Face 6

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]],self.M[:,row[3]]])

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection='3d')

        for item in [self.fig, ax]:
            item.patch.set_visible(False)

        self.facesObj = Poly3DCollection(faces, linewidths=.2, edgecolors='k',animated = True)
        self.facesObj.set_facecolor([(0,0,1,0.9), # Blue
        (0,1,0,0.9), # Green
        (.9,.5,0.13,0.9), # Orange
        (1,0,0,0.9), # Red
        (1,1,0,0.9), # Yellow
        (0,0,0,0.9)]) # Black

        # Transfering information to the plot
        ax.add_collection3d(self.facesObj)

        # Configuring the plot aspect
        ax.azim=-90
        ax.roll = 0
        ax.elev= 90   
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
        ax.set_aspect('equal')
        ax.disable_mouse_rotation()
        ax.set_axis_off()

        self.pix2unit = 1.0/60 # ratio for drawing the cube 

    def update_cube(self):
        """
        Updates the cube vertices and updates the figure.
        Call this function after modifying the vertex matrix in self.M to redraw the cube
        """
        faces = []
        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]], self.M[:,row[3]]])

        self.facesObj.set_verts(faces)
        self.bm.update()

    def canvas_coordinates_to_figure_coordinates(self,x_can,y_can):
        """
        Remap canvas coordinates to cube centered coordinates
        """

        (canvas_width,canvas_height)=self.canvas.get_width_height()
        figure_center_x = canvas_width/2+14
        figure_center_y = canvas_height/2+2
        x_fig = (x_can-figure_center_x)*self.pix2unit
        y_fig = (y_can-figure_center_y)*self.pix2unit

        return(x_fig,y_fig)

    def destroy(self):
        """
        Close function to properly destroy the window and tk with figure
        """
        try:
            self.destroy()
        finally:
            exit()


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
            cv.draw_idle()


if __name__ == "__main__":
    app = Arcball()
    app.mainloop()
    exit()
