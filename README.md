# driver-signature-raspberry-pi
Is it possible to predict who is driving a vehicle by analyzing the driving pattern of a driver? If there are multiple people driving a vehicle (mostly true in a commercial driving scenario), can the driving style be captured and learned?

The postulate is that this is possible - by reading data from the car's OBD port at a fast enough frequency. One can create feature sets from signals received from the car's CAN bus (via a ELM 327 interface) through the OBD port, then use that feature set to learn who the driver is. A real-time prediction is possible by deploying this model and quering it in real-time.

See the blog post at http://www.anupambagchi.com/blogs/data-science/item/69-driver-signatures-from-obd-data-captured-using-a-raspberry-pi-part-1-building-your-raspberry-pi-setup to get the details. This link takes you to a series of three articles that describes the gory details.

Some improvements to make this more accurate are mentioned in the blog posts mentioned above.
