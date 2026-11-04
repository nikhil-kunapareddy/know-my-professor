# knowmyprof.web.app

A one-page Firebase Hosting site that gives the Streamlit frontend a readable
address. It is a wrapper, not a proxy: `public/index.html` frames
`kmp-frontend` on Cloud Run.

## Why a frame and not a rewrite

Firebase Hosting can rewrite requests to Cloud Run, but it cannot carry
websockets, and Streamlit runs over one (`/_stcore/stream`). A rewrite would
load the page and then hang. A custom domain mapped straight onto Cloud Run
would avoid the frame, but costs a domain registration.

## Where it lives

- Firebase project **`know-my-professor-f4b98`**, a separate GCP project from
  `know-my-professor`. It is on the Spark plan with no billing account, so it
  cannot incur charges.
- Site **`knowmyprof`** → `https://knowmyprof.web.app` (also
  `knowmyprof.firebaseapp.com`).

## Deploy

Not in CI. It changes only if the frontend's Cloud Run URL changes, and that
URL appears twice: the iframe `src` in `public/index.html` and `frame-src` in
the CSP in `firebase.json`. Update both, or the browser blocks the frame.

```bash
cd deploy/hosting
firebase deploy --only hosting:knowmyprof
```
