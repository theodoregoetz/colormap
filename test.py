import wx

# tested on wxPython 2.8.11.0, Python 2.7.1+, Ubuntu 11.04
# http://stackoverflow.com/questions/2053268/side-effects-of-handling-evt-paint-event-in-wxpython
# http://stackoverflow.com/questions/25756896/drawing-to-panel-inside-of-frame-in-wxpython
# http://www.infinity77.net/pycon/tutorial/pyar/wxpython.html
# also, see: wx-2.8-gtk2-unicode/wx/lib/agw/buttonpanel.py

class MyPanel(wx.Panel): #(wx.PyPanel): #PyPanel also works
  def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.DefaultSize, style=0, name="MyPanel"):
    super(MyPanel, self).__init__(parent, id, pos, size, style, name)
    self.Bind(wx.EVT_SIZE, self.OnSize)
    self.Bind(wx.EVT_PAINT, self.OnPaint)
  def OnSize(self, event):
    print("OnSize" +str(event))
    #self.SetClientRect(event.GetRect()) # no need
    self.Refresh() # MUST have this, else the rectangle gets rendered corruptly when resizing the window!
    event.Skip() # seems to reduce the ammount of OnSize and OnPaint events generated when resizing the window
  def OnPaint(self, event):
    #~ dc = wx.BufferedPaintDC(self) # works, somewhat
    dc = wx.PaintDC(self) # works
    print(dc)
    rect = self.GetClientRect()
    # "Set a red brush to draw a rectangle"
    dc.SetBrush(wx.RED_BRUSH)
    dc.DrawRectangle(10, 10, rect[2]-20, 50)
    #self.Refresh() # recurses here!


class MyFrame(wx.Frame):
  def __init__(self, parent):
    wx.Frame.__init__(self, parent, -1, "Custom Panel Demo")
    self.SetSize((300, 200))
    self.panel = MyPanel(self) #wx.Panel(self)
    self.panel.SetBackgroundColour(wx.Colour(10,10,10))
    self.panel.SetForegroundColour(wx.Colour(50,50,50))
    sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
    sizer_1.Add(self.panel, 1, wx.EXPAND | wx.ALL, 0)
    self.SetSizer(sizer_1)
    self.Layout()

app = wx.App(0)
frame = MyFrame(None)
app.SetTopWindow(frame)
frame.Show()
app.MainLoop()
