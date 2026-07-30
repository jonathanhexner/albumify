# Narration — People & Faces: blank filter and missing face boxes

Here's the problem in plain terms.

In Albumify, you open the People and Faces page, click on a person, and you see all
their photos. Two things are broken. The "Selected" filter shows nothing, and the
image-detail popup never draws boxes around the faces.

The symptom looks like two separate bugs, but it's one cause.

When the app asks the server for a person's photos, the server replies with only three
facts about each photo: its filename, how many faces it has, and the list of faces.
That's it. Whether the photo was selected, its quality score, and the coordinates of
each face box — none of that is included in the reply.

Think of a paper form with only three blank lines. Anything that isn't on the form gets
thrown away before it reaches the screen. So when the page asks "was this photo
selected?" or "where are the faces?", the answer comes back blank. The filter matches
nothing, and the boxes have no coordinates to draw.

The proof is simple. The database clearly holds the selection flag and the face box
coordinates for every photo. But the one function that lists a person's photos copies
only those three fields and never reads the rest.

The good news: the team already built the fix for a nearly identical bug elsewhere — a
single shared shape for all per-image data. This page just needs to use it.

---
Run through any TTS for audio (e.g. PowerShell `System.Speech`).
