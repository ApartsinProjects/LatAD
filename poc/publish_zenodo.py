"""Deposit the LatAD checkpoint bundle to Zenodo using YOUR token.

The token is read from the environment variable ZENODO_TOKEN (never hard-coded, never
passed as an argument that lands in shell history). Get one at
https://zenodo.org/account/settings/applications/tokens/new/ with scopes
'deposit:write' and 'deposit:actions'.

Steps performed:
  1. create a new deposition (reserves a DOI)
  2. upload zenodo_LatAD_v1.0.zip
  3. attach metadata from zenodo_bundle/zenodo.json
  4. print the reserved DOI and the edit URL
  5. PUBLISH -- only if you pass --publish (this is PERMANENT and cannot be undone)

Usage:
  export ZENODO_TOKEN=...            # your personal token (bash);  $env:ZENODO_TOKEN='...' in PowerShell
  python publish_zenodo.py           # dry deposit: uploads + metadata + reserved DOI, does NOT publish
  python publish_zenodo.py --publish # finalize (irreversible)
  python publish_zenodo.py --sandbox # test against sandbox.zenodo.org first (recommended)
"""
import os, sys, json, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "zenodo_LatAD_v1.0.zip")
META = os.path.join(HERE, "zenodo_bundle", "zenodo.json")
SANDBOX = "--sandbox" in sys.argv
PUBLISH = "--publish" in sys.argv
BASE = "https://sandbox.zenodo.org/api" if SANDBOX else "https://zenodo.org/api"


def token():
    t = os.environ.get("ZENODO_TOKEN")
    if not t:
        sys.exit("ERROR: set ZENODO_TOKEN in your environment first (see the file header).")
    return t


def req(method, url, data=None, headers=None, raw=False):
    h = {"Authorization": f"Bearer {token()}"}
    if headers:
        h.update(headers)
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            body = resp.read()
            return json.loads(body) if not raw else body
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {url}\n{e.read().decode(errors='ignore')}")


def main():
    if not os.path.exists(ZIP):
        sys.exit(f"missing {ZIP} -- build it first (make_archive of zenodo_bundle).")
    print(f"Target: {BASE}   publish={PUBLISH}")

    dep = req("POST", f"{BASE}/deposit/depositions",
              data=b"{}", headers={"Content-Type": "application/json"})
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    reserved = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(reserved on save)")
    print(f"  deposition id: {dep_id}")
    print(f"  reserved DOI : {reserved}")

    with open(ZIP, "rb") as f:
        req("PUT", f"{bucket}/{os.path.basename(ZIP)}", data=f.read(),
            headers={"Content-Type": "application/octet-stream"})
    print(f"  uploaded     : {os.path.basename(ZIP)}")

    meta = json.load(open(META, encoding="utf-8"))
    req("PUT", f"{BASE}/deposit/depositions/{dep_id}",
        data=json.dumps(meta).encode(), headers={"Content-Type": "application/json"})
    print("  metadata     : attached")

    edit_url = dep["links"].get("html", f"{BASE.replace('/api','')}/deposit/{dep_id}")
    if PUBLISH:
        out = req("POST", f"{BASE}/deposit/depositions/{dep_id}/actions/publish")
        print(f"  PUBLISHED    : {out.get('doi_url', out.get('doi'))}")
    else:
        print(f"  NOT published. Review/finalize here: {edit_url}")
        print(f"  Re-run with --publish to finalize (permanent), or click Publish on that page.")
    print(f"\n  Reserved/record DOI to paste into the paper: {reserved}")


if __name__ == "__main__":
    main()
