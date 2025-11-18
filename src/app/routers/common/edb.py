from fastapi import APIRouter

from ...database.dm_pool import dm_pool

router = APIRouter()

@router.get("/", tags=["edb"])
async def test():
    return {"message": "Hello, edb!"}


@router.get("/user/{user_no}", tags=["yus"])
async def get_user(user_no: int):
    select_user_query = "SELECT * FROM h_org_user WHERE employeeNo = ?"
    user = await dm_pool.fetch_one(select_user_query,user_no)
    return {"message": f"Hello, {user_no}!", "user": user}