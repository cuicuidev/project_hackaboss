import streamlit as st

class Section:

    def __init__(self, text, title = None, header = None, media = None, media_on_left = False, flex = (5, 4)):
        self.text = text
        self.title = title
        self.header = header
        self.media_callback = media  # This is now a callback/lambda
        self.media_on_left = media_on_left
        self.flex = flex

    def render(self):
        if self._hasMedia():
            col1, col2 = st.columns(self.flex)[::-1] if self.media_on_left else st.columns(self.flex)

            if self._hasTitle():
                col1.title(self.title)

            if self._hasHeader():
                col1.header(self.header)

            col1.write(self.text)
            self._renderMedia(col2)

        else:

            if self._hasTitle():
                st.title(self.title)

            if self._hasHeader():
                st.header(self.header)

            st.write(self.text)

    def _renderMedia(self, column):
        if self.media_callback:
            self.media_callback(column)

    def _hasMedia(self):
        return self.media_callback is not None

    def _hasTitle(self):
        return self.title is not None
    
    def _hasHeader(self):
        return self.header is not None